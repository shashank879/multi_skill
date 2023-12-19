import functools

import embodied
import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd

from . import agent
from . import expl
from . import nets
from . import tfutils

from .hierarchy import Hierarchy


class GPA(Hierarchy):

    def __init__(self, wm, act_space, config):
        self.wm = wm
        self.config = config
        self.extr_reward = lambda traj: self.wm.heads['reward'](traj).mean()[1:]
        self.skill_shape = config.gpa.skill_vae.skill_shape
        self.state_shape = (self.config.rssm.deter,)

        self.skill_vae = agent.MultiSkillVAE(self.state_shape, self.skill_shape, config.gpa.skill_vae)

        # Worker
        wconfig = config.update({
            'actor.inputs': self.config.worker_inputs,
            'critic.inputs': self.config.worker_inputs,
        })
        self.worker = agent.ImagActorCritic({
            'extr': agent.VFunction(lambda s: s['reward_extr'], wconfig),
            'expl': agent.VFunction(lambda s: s['reward_expl'], wconfig),
            'goal': agent.VFunction(lambda s: s['reward_goal'], wconfig),
        }, config.worker_rews, act_space, wconfig)

        # Manager
        mconfig = config.update({
            'actor_grad_cont': 'reinforce',
            'actent.target': config.manager_actent,
        })
        self.manager = agent.MultiSkillImagActorCritic({
            'extr': agent.VFunction(lambda s: s['reward_extr'], mconfig),
            'expl': agent.VFunction(lambda s: s['reward_expl'], mconfig),
            'goal': agent.VFunction(lambda s: s['reward_goal'], mconfig),
        }, config.manager_rews, config.gpa.skill_vae.skill_shape, self.skill_vae.num_vaes, mconfig)

        if self.config.expl_rew == 'disag':
            self.expl_reward = expl.Disag(wm, act_space, config)
        elif self.config.expl_rew == 'adver':
            self.expl_reward = self.elbo_reward
        else:
            raise NotImplementedError(self.config.expl_rew)
        if config.explorer:
            self.explorer = agent.ImagActorCritic({
                'expl': agent.VFunction(self.expl_reward, config),
            }, {'expl': 1.0}, act_space, config)

        self.feat = nets.Input(['deter'])
        # self.goal_enc = nets.MLP(
        #     self.goal_enc_shape, dims='context', **config.gpa.goal_gen.encoder)
        # self.goal_dec = nets.MLP(
        #     self.state_shape, dims='context', **config.gpa.goal_gen.decoder)
        # self.goal_prior = tfutils.get_prior(config.gpa.goal_gen.encoder, self.goal_enc_shape)
        # self.goal_kl = tfutils.AutoAdapt((), **config.encdec_kl)
        # self.goal_opt = tfutils.Optimizer('goal', **config.encdec_opt)

    def initial(self, batch_size):
        return {
            'step': tf.zeros((batch_size,), tf.int64),
            'skill': tf.zeros((batch_size, self.skill_vae.num_vaes,) + self.skill_shape, tf.float32),
            'choice': tf.zeros((batch_size, self.skill_vae.num_vaes,), tf.float32),
            'goal': tf.zeros((batch_size,) + self.state_shape, tf.float32),
            'goal_opts': tf.zeros((batch_size, self.skill_vae.num_vaes, self.config.rssm.deter,), tf.float32),
            'goal_opts_stoch': tf.zeros((batch_size, self.skill_vae.num_vaes, self.config.rssm.stoch, self.config.rssm.classes), tf.float32),
        }

    def policy(self, latent, carry, imag=False):
        duration = self.config.train_skill_duration if imag else (
            self.config.env_skill_duration)
        sg = lambda x: tf.nest.map_structure(tf.stop_gradient, x)
        update = (carry['step'] % duration) == 0
        switch = lambda x, y: (
            tf.einsum('i,i...->i...', 1 - update.astype(x.dtype), x) +
            tf.einsum('i,i...->i...', update.astype(x.dtype), y))

        # Manager policy (state -> worker_goal)
        manager_action = self.manager.policy(sg(latent))
        skill = sg(switch(carry['skill'], manager_action['skill'].sample()))
        new_goal_opts = self.skill_vae.decode({'skill': skill, 'context': self.feat(latent)}).mode()

        if self.config.gpa.manager.act_biased_choice and (not imag):
            choice_bias = manager_action['choice'].probs_parameter()
            new_choice = self.choose_goal(new_goal_opts, choice_bias)
        else:
            new_choice = manager_action['choice'].sample()
            # new_choice = self.choose_goal(new_goal_opts)
        choice = sg(switch(carry['choice'], new_choice))

        new_goal = tf.reduce_sum(tf.expand_dims(choice, axis=-1) * tf.stop_gradient(new_goal_opts), axis=-2)
        new_goal = (
            self.feat(latent).astype(tf.float32) + new_goal if self.config.manager_delta else new_goal)
        goal = sg(switch(carry['goal'], new_goal))

        delta = goal - self.feat(latent).astype(tf.float32)
        dist = self.worker.actor(sg({**latent, 'goal': goal, 'delta': delta}))
        outs = {'action': dist}
        if 'image' in self.wm.heads['decoder'].shapes:
            outs['log_goal'] = self.wm.heads['decoder']({
                'deter': goal, 'stoch': self.wm.rssm.get_stoch(goal),
            })['image'].mode()
        carry = {'step': carry['step'] + 1, 'skill': skill, 'choice': choice, 'goal': goal, 'goal_opts': new_goal_opts, 'goal_opts_stoch': self.wm.rssm.get_stoch(new_goal_opts)}
        return outs, carry

    def choose_goal(self, goal_opts, choice_bias=None):
        # vae_act_prob = self.skill_vae.act_prior.prob(skills)
        state = {'deter': goal_opts,
                 'stoch': self.wm.rssm.get_stoch(goal_opts),}
        scores = []
        for key, critic in self.manager.critics.items():
            ret = critic.target_net(state)
            ret = self.manager.retnorms[key](ret.mean())
            scores.append(ret * self.manager.scales[key])
        score = tf.reduce_sum(scores, 0)
        # score = tf.stack(scores, axis=-1)
        if not choice_bias is None:
            score += choice_bias
        probs = tfutils.OneHotDist(logits=score)
        choice = probs.sample()
        return choice

    def train_jointly(self, imagine, start):
        start = start.copy()
        metrics = {}
        with tf.GradientTape(persistent=True) as tape:
            policy = functools.partial(self.policy, imag=True)
            traj = self.wm.imagine_carry(
                policy, start, self.config.imag_horizon,
                self.initial(len(start['is_first'])))
            traj['reward_extr'] = self.extr_reward(traj)
            traj['reward_expl'] = self.expl_reward(traj)
            traj['reward_goal'] = self.goal_reward(traj)
            traj['delta'] = traj['goal'] - self.feat(traj).astype(tf.float32)
            wtraj = self.split_traj(traj)
            mtraj = self.abstract_traj(traj)
        mets = self.worker.update(wtraj, tape)
        metrics.update({f'worker_{k}': v for k, v in mets.items()})
        mets = self.manager.update(mtraj, tape)
        metrics.update({f'manager_{k}': v for k, v in mets.items()})
        del tape
        return traj, metrics

    def train_manager(self, imagine, start):
        start = start.copy()
        metrics = {}
        with tf.GradientTape(persistent=True) as tape:
            policy = functools.partial(self.policy, imag=True)
            traj = self.wm.imagine_carry(
                policy, start, self.config.imag_horizon,
                self.initial(len(start['is_first'])))
            traj['reward_extr'] = self.extr_reward(traj)
            traj['reward_expl'] = self.expl_reward(traj)
            traj['reward_goal'] = self.goal_reward(traj)
            traj['delta'] = traj['goal'] - self.feat(traj).astype(tf.float32)
            mtraj = self.abstract_traj(traj)
        mets = self.manager.update(mtraj, tape)
        metrics.update({f'manager_{k}': v for k, v in mets.items()})
        del tape
        return traj, metrics

    def train_vae_replay(self, data):
        traj = tf.nest.map_structure(lambda x: tf.transpose(x, perm=[1, 0] + list(range(2, len(x.shape)))), data)
        return self.train_vae(traj)

    def train_vae_imag(self, traj):
        return self.train_vae(traj)

    def train_vae(self, traj):
        metrics = {}
        mets = self.skill_vae.train(traj)
        metrics.update({'skvae_'+k:v for k,v in mets.items()})
        return metrics

    def propose_goal(self, start, impl):
        feat = self.feat(start).astype(tf.float32)
        if impl == 'replay':
            target = tf.random.shuffle(feat).astype(tf.float32)
            skill = self.skill_vae.encode({'goal': target, 'context': feat}).sample()
            goal_opts = self.skill_vae.decode({'skill': skill, 'context': feat}).mode()
            choice = self.choose_goal(goal_opts)
            return tf.reduce_sum(tf.expand_dims(choice, axis=-1) * goal_opts, axis=-2)
        if impl == 'replay_direct':
            return tf.random.shuffle(feat).astype(tf.float32)
        if impl == 'manager':
            manager_action = self.manager.policy(start)
            goal_opts = self.skill_vae.decode({'skill': manager_action['skill'].sample(), 'context': feat}).mode()
            goal = tf.reduce_sum(tf.expand_dims(manager_action['choice'].sample(), axis=-1) * goal_opts, axis=-2)
            goal = feat + goal if self.config.manager_delta else goal
            return goal
        if impl == 'prior':
            skill = self.skill_vae.prior.sample(start['is_terminal'].shape)
            goal_opts = self.skill_vae.decode({'skill': skill, 'context': feat}).mode()
            choice = self.choose_goal(goal_opts)
            return tf.reduce_sum(tf.expand_dims(choice, axis=-1) * goal_opts, axis=-2)
        raise NotImplementedError(impl)

    def goal_reward(self, traj):
        feat = self.feat(traj).astype(tf.float32)
        goal = tf.stop_gradient(traj['goal'].astype(tf.float32))
        skill = tf.stop_gradient(traj['skill'].astype(tf.float32))
        context = tf.stop_gradient(
            tf.repeat(feat[0][None], 1 + self.config.imag_horizon, 0))
        if self.config.goal_reward == 'dot':
            return tf.einsum('...i,...i->...', goal, feat)[1:]
        elif self.config.goal_reward == 'dir':
            return tf.einsum(
                '...i,...i->...', tf.nn.l2_normalize(goal, -1), feat)[1:]
        elif self.config.goal_reward == 'normed_inner':
            norm = tf.linalg.norm(goal, axis=-1, keepdims=True)
            return tf.einsum('...i,...i->...', goal / norm, feat / norm)[1:]
        elif self.config.goal_reward == 'normed_squared':
            norm = tf.linalg.norm(goal, axis=-1, keepdims=True)
            return -((goal / norm - feat / norm) ** 2).mean(-1)[1:]
        elif self.config.goal_reward == 'cosine_lower':
            gnorm = tf.linalg.norm(goal, axis=-1, keepdims=True) + 1e-12
            fnorm = tf.linalg.norm(feat, axis=-1, keepdims=True) + 1e-12
            fnorm = tf.maximum(gnorm, fnorm)
            return tf.einsum('...i,...i->...', goal / gnorm, feat / fnorm)[1:]
        elif self.config.goal_reward == 'cosine_lower_pos':
            gnorm = tf.linalg.norm(goal, axis=-1, keepdims=True) + 1e-12
            fnorm = tf.linalg.norm(feat, axis=-1, keepdims=True) + 1e-12
            fnorm = tf.maximum(gnorm, fnorm)
            cos = tf.einsum('...i,...i->...', goal / gnorm, feat / fnorm)[1:]
            return tf.nn.relu(cos)
        elif self.config.goal_reward == 'cosine_frac':
            gnorm = tf.linalg.norm(goal, axis=-1) + 1e-12
            fnorm = tf.linalg.norm(feat, axis=-1) + 1e-12
            goal /= gnorm[..., None]
            feat /= fnorm[..., None]
            cos = tf.einsum('...i,...i->...', goal, feat)
            mag = tf.minimum(gnorm, fnorm) / tf.maximum(gnorm, fnorm)
            return (cos * mag)[1:]
        elif self.config.goal_reward == 'cosine_frac_pos':
            gnorm = tf.linalg.norm(goal, axis=-1) + 1e-12
            fnorm = tf.linalg.norm(feat, axis=-1) + 1e-12
            goal /= gnorm[..., None]
            feat /= fnorm[..., None]
            cos = tf.einsum('...i,...i->...', goal, feat)
            mag = tf.minimum(gnorm, fnorm) / tf.maximum(gnorm, fnorm)
            return tf.nn.relu(cos * mag)[1:]
        elif self.config.goal_reward == 'cosine_max':
            gnorm = tf.linalg.norm(goal, axis=-1, keepdims=True) + 1e-12
            fnorm = tf.linalg.norm(feat, axis=-1, keepdims=True) + 1e-12
            norm = tf.maximum(gnorm, fnorm)
            return tf.einsum('...i,...i->...', goal / norm, feat / norm)[1:]
        elif self.config.goal_reward == 'cosine_max_pos':
            gnorm = tf.linalg.norm(goal, axis=-1, keepdims=True) + 1e-12
            fnorm = tf.linalg.norm(feat, axis=-1, keepdims=True) + 1e-12
            norm = tf.maximum(gnorm, fnorm)
            cos = tf.einsum('...i,...i->...', goal / norm, feat / norm)[1:]
            return tf.nn.relu(cos)
        elif self.config.goal_reward == 'normed_inner_clip':
            norm = tf.linalg.norm(goal, axis=-1, keepdims=True)
            cosine = tf.einsum('...i,...i->...', goal / norm, feat / norm)[1:]
            return tf.clip_by_value(cosine, -1.0, 1.0)
        elif self.config.goal_reward == 'normed_inner_clip_pos':
            norm = tf.linalg.norm(goal, axis=-1, keepdims=True)
            cosine = tf.einsum('...i,...i->...', goal / norm, feat / norm)[1:]
            return tf.clip_by_value(cosine, 0.0, 1.0)
        elif self.config.goal_reward == 'diff':
            goal = tf.nn.l2_normalize(goal[:-1], -1)
            diff = tf.concat([feat[1:] - feat[:-1]], 0)
            return tf.einsum('...i,...i->...', goal, diff)
        elif self.config.goal_reward == 'norm':
            return -tf.linalg.norm(goal - feat, axis=-1)[1:]
        elif self.config.goal_reward == 'squared':
            return -((goal - feat) ** 2).sum(-1)[1:]
        elif self.config.goal_reward == 'epsilon':
            return ((goal - feat).mean(-1) < 1e-3).astype(tf.float32)[1:]
        elif self.config.goal_reward == 'enclogprob':
            return self.skill_vae.encode({'goal': goal, 'context': context}).log_prob(skill)[1:]
        elif self.config.goal_reward == 'encprob':
            return self.skill_vae.encode({'goal': goal, 'context': context}).prob(skill)[1:]
        elif self.config.goal_reward == 'enc_normed_cos':
            dist = self.skill_vae.encode({'goal': goal, 'context': context})
            probs = dist.distribution.probs_parameter()
            norm = tf.linalg.norm(probs, axis=[-2, -1], keepdims=True)
            return tf.einsum('...ij,...ij->...', probs / norm, skill / norm)[1:]
        elif self.config.goal_reward == 'enc_normed_squared':
            dist = self.skill_vae.encode({'goal': goal, 'context': context})
            probs = dist.distribution.probs_parameter()
            norm = tf.linalg.norm(probs, axis=[-2, -1], keepdims=True)
            return -((probs / norm - skill / norm) ** 2).mean([-2, -1])[1:]
        else:
            raise NotImplementedError(self.config.goal_reward)

    def elbo_reward(self, traj):
        feat = self.feat(traj).astype(tf.float32)
        context = tf.repeat(feat[0][None], 1 + self.config.imag_horizon, 0)
        enc = self.skill_vae.encode({'goal': feat, 'context': context})
        dec = self.skill_vae.decode({'skill': enc.sample(), 'context': context})
        feat_stack = tf.stack([feat] * self.skill_vae.num_vaes, axis=len(context.shape)-1)
        ll = dec.log_prob(feat_stack)
        kl = enc.kl_divergence(self.skill_vae.act_prior)
        if self.config.adver_impl == 'abs':
            return tf.abs(dec.mode() - feat_stack).mean(-1).min(-1)[1:]
        elif self.config.adver_impl == 'squared':
            return ((dec.mode() - feat_stack) ** 2).mean(-1).min(-1)[1:]
        elif self.config.adver_impl == 'elbo_scaled':
            scales = tf.stack([kl_norm.scale() for kl_norm in self.skill_vae.kl_norms], -1)
            scales = scales[None, ...][None, ...]
            return (kl - ll / scales).min(-1)[1:]
        elif self.config.adver_impl == 'elbo_unscaled':
            return (kl - ll).min(-1)[1:]
        raise NotImplementedError(self.config.adver_impl)

    def abstract_traj(self, traj):
        traj = traj.copy()
        k = self.config.train_skill_duration
        reshape = lambda x: x.reshape([x.shape[0] // k, k] + x.shape[1:])
        weights = tf.math.cumprod(reshape(traj['cont'][:-1]), 1)
        for key, value in list(traj.items()):
            if 'reward' in key:
                traj[key] = (reshape(value) * weights).mean(1)
            elif key == 'cont':
                traj[key] = tf.concat([value[:1], reshape(value[1:]).prod(1)], 0)
            else:
                traj[key] = tf.concat([reshape(value[:-1])[:, 0], value[-1:]], 0)
        traj['weight'] = tf.math.cumprod(
            self.config.discount * traj['cont']) / self.config.discount
        return traj

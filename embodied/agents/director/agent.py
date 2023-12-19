import sys

import embodied
import ruamel.yaml as yaml
import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision as prec
from tensorflow_probability import distributions as tfd

from . import behaviors
from . import nets
from . import tfagent
from . import tfutils


class Agent(tfagent.TFAgent):

  configs = yaml.YAML(typ='safe').load((
      embodied.Path(sys.argv[0]).parent / 'configs.yaml').read())

  def __init__(self, obs_space, act_space, step, config):
    self.config = config
    self.obs_space = obs_space
    self.act_space = act_space['action']
    self.step = step
    self.wm = WorldModel(obs_space, config)
    self.task_behavior = getattr(behaviors, config.task_behavior)(
        self.wm, self.act_space, self.config)
    if config.expl_behavior == 'None':
      self.expl_behavior = self.task_behavior
    else:
      self.expl_behavior = getattr(behaviors, config.expl_behavior)(
          self.wm, self.act_space, self.config)
    self.initial_policy_state = tf.function(lambda obs: (
        self.wm.rssm.initial(len(obs['is_first'])),
        self.task_behavior.initial(len(obs['is_first'])),
        self.expl_behavior.initial(len(obs['is_first'])),
        tf.zeros((len(obs['is_first']),) + self.act_space.shape)))
    self.initial_train_state = tf.function(lambda obs: (
        self.wm.rssm.initial(len(obs['is_first']))))

  @tf.function
  def policy(self, obs, state=None, mode='train'):
    self.config.tf.jit and print('Tracing policy function.')
    if state is None:
      state = self.initial_policy_state(obs)
    obs = self.preprocess(obs)
    latent, task_state, expl_state, action = state
    embed = self.wm.encoder(obs)
    latent, _ = self.wm.rssm.obs_step(latent, action, embed, obs['is_first'])
    noise = self.config.expl_noise
    if mode == 'eval':
      noise = self.config.eval_noise
      outs, task_state = self.task_behavior.policy(latent, task_state)
      outs = {**outs, 'action': outs['action'].mode()}
    elif mode == 'explore':
      outs, expl_state = self.expl_behavior.policy(latent, expl_state)
      outs = {**outs, 'action': outs['action'].sample()}
    elif mode == 'train':
      outs, task_state = self.task_behavior.policy(latent, task_state)
      outs = {**outs, 'action': outs['action'].sample()}
    outs = {**outs, 'action': tfutils.action_noise(
        outs['action'], noise, self.act_space)}
    state = (latent, task_state, expl_state, outs['action'])
    return outs, state

  @tf.function
  def train(self, data, state=None):
    self.config.tf.jit and print('Tracing train function.')
    metrics = {}
    if state is None:
      state = self.initial_train_state(data)
    data = self.preprocess(data)
    state, wm_outs, mets = self.wm.train(data, state)
    metrics.update(mets)
    context = {**data, **wm_outs['post']}
    start = tf.nest.map_structure(
        lambda x: x.reshape([-1] + list(x.shape[2:])), context)
    _, mets = self.task_behavior.train(self.wm.imagine, start, context)
    metrics.update(mets)
    if self.config.expl_behavior != 'None':
      _, mets = self.expl_behavior.train(self.wm.imagine, start, context)
      metrics.update({'expl_' + key: value for key, value in mets.items()})
    outs = {}
    if 'key' in data:
      criteria = {**data, **wm_outs}
      outs.update(key=data['key'], priority=criteria[self.config.priority])
    return outs, state, metrics

  @tf.function
  def report(self, data):
    self.config.tf.jit and print('Tracing report function.')
    data = self.preprocess(data)
    report = {}
    report.update(self.wm.report(data))
    mets = self.task_behavior.report(data)
    report.update({f'task_{k}': v for k, v in mets.items()})
    if self.expl_behavior is not self.task_behavior:
      mets = self.expl_behavior.report(data)
      report.update({f'expl_{k}': v for k, v in mets.items()})
    return report

  def dataset(self, generator):
    if self.config.data_loader == 'tfdata':
      example = next(generator())
      dtypes = {k: v.dtype for k, v in example.items()}
      shapes = {k: v.shape for k, v in example.items()}
      return tf.data.Dataset.range(self.config.batch_size).interleave(
          lambda _: tf.data.Dataset.from_generator(generator, dtypes, shapes),
      ).batch(self.config.batch_size).prefetch(tf.data.AUTOTUNE)
    elif self.config.data_loader == 'embodied':
      return embodied.Prefetch(
          sources=[generator] * self.config.batch_size,
          workers=8, prefetch=4)
    else:
      raise NotImplementedError(self.config.data_loader)

  def preprocess(self, obs):
    dtype = prec.global_policy().compute_dtype
    obs = {k: tf.tensor(v) for k, v in obs.items()}
    obs = obs.copy()
    for key, value in obs.items():
      if key.startswith('log_') or key in ('key',):
        continue
      if len(value.shape) > 3 and value.dtype == tf.uint8:
        value = value.astype(dtype) / 255.0
      else:
        value = value.astype(tf.float32)
      obs[key] = value
    obs['reward'] = {
        'off': tf.identity, 'sign': tf.sign,
        'tanh': tf.tanh, 'symlog': tfutils.symlog,
    }[self.config.transform_rewards](obs['reward'])
    obs['cont'] = 1.0 - obs['is_terminal'].astype(tf.float32)
    return obs


class WorldModel(tfutils.Module):

  def __init__(self, obs_space, config):
    shapes = {k: tuple(v.shape) for k, v in obs_space.items()}
    shapes = {k: v for k, v in shapes.items() if not k.startswith('log_')}
    self.config = config
    self.rssm = nets.RSSM(**config.rssm)
    self.encoder = nets.MultiEncoder(shapes, **config.encoder)
    self.heads = {}
    self.heads['decoder'] = nets.MultiDecoder(shapes, **config.decoder)
    self.heads['reward'] = nets.MLP((), **config.reward_head)
    self.heads['cont'] = nets.MLP((), **config.cont_head)
    self.model_opt = tfutils.Optimizer('model', **config.model_opt)
    self.wmkl = tfutils.AutoAdapt((), **self.config.wmkl, inverse=False)

  def train(self, data, state=None):
    with tf.GradientTape() as model_tape:
      model_loss, state, outputs, metrics = self.loss(
          data, state, training=True)
    modules = [self.encoder, self.rssm, *self.heads.values()]
    metrics.update(self.model_opt(model_tape, model_loss, modules))
    return state, outputs, metrics

  def loss(self, data, state=None, training=False):
    metrics = {}
    embed = self.encoder(data)
    post, prior = self.rssm.observe(
        embed, data['action'], data['is_first'], state)
    dists = {}
    post_const = tf.nest.map_structure(tf.stop_gradient, post)
    for name, head in self.heads.items():
      out = head(post if name in self.config.grad_heads else post_const)
      if not isinstance(out, dict):
        out = {name: out}
      dists.update(out)
    losses = {}
    kl = self.rssm.kl_loss(post, prior, self.config.wmkl_balance)
    kl, mets = self.wmkl(kl, update=training)
    losses['kl'] = kl
    metrics.update({f'wmkl_{k}': v for k, v in mets.items()})
    for key, dist in dists.items():
      losses[key] = -dist.log_prob(data[key].astype(tf.float32))
    metrics.update({f'{k}_loss_mean': v.mean() for k, v in losses.items()})
    metrics.update({f'{k}_loss_std': v.std() for k, v in losses.items()})
    scaled = {}
    for key, loss in losses.items():
      assert loss.shape == embed.shape[:2], (key, loss.shape)
      scaled[key] = loss * self.config.loss_scales.get(key, 1.0)
    model_loss = sum(scaled.values())
    if 'prob' in data and self.config.priority_correct:
      weights = (1.0 / data['prob']) ** self.config.priority_correct
      weights /= weights.max()
      assert weights.shape == model_loss.shape
      model_loss *= weights
    out = {'embed':  embed, 'post': post, 'prior': prior}
    out.update({f'{k}_loss': v for k, v in losses.items()})
    metrics['prior_ent'] = self.rssm.get_dist(prior).entropy().mean()
    metrics['post_ent'] = self.rssm.get_dist(post).entropy().mean()
    metrics['model_loss_mean'] = model_loss.mean()
    metrics['model_loss_std'] = model_loss.std()
    if not self.config.tf.debug_nans:
      if 'reward' in dists:
        stats = tfutils.balance_stats(dists['reward'], data['reward'], 0.1)
        metrics.update({f'reward_{k}': v for k, v in stats.items()})
      if 'cont' in dists:
        stats = tfutils.balance_stats(dists['cont'], data['cont'], 0.5)
        metrics.update({f'cont_{k}': v for k, v in stats.items()})
    last_state = {k: v[:, -1] for k, v in post.items()}
    return model_loss.mean(), last_state, out, metrics

  def imagine(self, policy, start, horizon):
    first_cont = (1.0 - start['is_terminal']).astype(tf.float32)
    keys = list(self.rssm.initial(1).keys())
    start = {k: v for k, v in start.items() if k in keys}
    start['action'] = policy(start)
    def step(prev, _):
      prev = prev.copy()
      action = prev.pop('action')
      state = self.rssm.img_step(prev, action)
      action = policy(state)
      return {**state, 'action': action}
    traj = tfutils.scan(
        step, tf.range(horizon), start, self.config.imag_unroll)
    traj = {k: tf.concat([start[k][None], v], 0) for k, v in traj.items()}
    traj['cont'] = tf.concat([
        first_cont[None], self.heads['cont'](traj).mean()[1:]], 0)
    traj['weight'] = tf.math.cumprod(
        self.config.discount * traj['cont']) / self.config.discount
    return traj

  def imagine_carry(self, policy, start, horizon, carry):
    first_cont = (1.0 - start['is_terminal']).astype(tf.float32)
    keys = list(self.rssm.initial(1).keys())
    start = {k: v for k, v in start.items() if k in keys}
    keys += list(carry.keys()) + ['action']
    states = [start]
    outs, carry = policy(start, carry)
    action = outs['action']
    if hasattr(action, 'sample'):
      action = action.sample()
    actions = [action]
    carries = [carry]
    for _ in range(horizon):
      states.append(self.rssm.img_step(states[-1], actions[-1]))
      outs, carry = policy(states[-1], carry)
      action = outs['action']
      if hasattr(action, 'sample'):
        action = action.sample()
      actions.append(action)
      carries.append(carry)
    transp = lambda x: {k: [x[t][k] for t in range(len(x))] for k in x[0]}
    traj = {**transp(states), **transp(carries), 'action': actions}
    traj = {k: tf.stack(v, 0) for k, v in traj.items()}
    cont = self.heads['cont'](traj).mean()
    cont = tf.concat([first_cont[None], cont[1:]], 0)
    traj['cont'] = cont
    traj['weight'] = tf.math.cumprod(
        self.config.imag_discount * cont) / self.config.imag_discount
    return traj

  def report(self, data):
    report = {}
    report.update(self.loss(data)[-1])
    context, _ = self.rssm.observe(
        self.encoder(data)[:6, :5], data['action'][:6, :5],
        data['is_first'][:6, :5])
    start = {k: v[:, -1] for k, v in context.items()}
    recon = self.heads['decoder'](context)
    openl = self.heads['decoder'](
        self.rssm.imagine(data['action'][:6, 5:], start))
    for key in self.heads['decoder'].cnn_shapes.keys():
      truth = data[key][:6].astype(tf.float32)
      model = tf.concat([recon[key].mode()[:, :5], openl[key].mode()], 1)
      error = (model - truth + 1) / 2
      video = tf.concat([truth, model, error], 2)
      report[f'openl_{key}'] = tfutils.video_grid(video)
    return report


class ImagActorCritic(tfutils.Module):

  def __init__(self, critics, scales, act_space, config):
    critics = {k: v for k, v in critics.items() if scales[k]}
    for key, scale in scales.items():
      assert not scale or key in critics, key
    self.critics = {k: v for k, v in critics.items() if scales[k]}
    self.scales = scales
    self.act_space = act_space
    self.config = config
    self.actor = nets.MLP(act_space.shape, **self.config.actor, dist=(
        config.actor_dist_disc if act_space.discrete
        else config.actor_dist_cont))
    self.grad = (
        config.actor_grad_disc if act_space.discrete
        else config.actor_grad_cont)
    self.advnorm = tfutils.Normalize(**self.config.advnorm)
    self.retnorms = {
        k: tfutils.Normalize(**self.config.retnorm) for k in self.critics}
    self.scorenorms = {
        k: tfutils.Normalize(**self.config.scorenorm) for k in self.critics}
    if self.config.actent_perdim:
      shape = act_space.shape[:-1] if act_space.discrete else act_space.shape
      self.actent = tfutils.AutoAdapt(
          shape, **self.config.actent, inverse=True)
    else:
      self.actent = tfutils.AutoAdapt((), **self.config.actent, inverse=True)
    self.opt = tfutils.Optimizer('actor', **self.config.actor_opt)

  def initial(self, batch_size):
    return None

  def policy(self, state, carry):
    return {'action': self.actor(state)}, carry

  def train(self, imagine, start, context):
    policy = lambda s: self.actor(tf.nest.map_structure(
        tf.stop_gradient, s)).sample()
    with tf.GradientTape(persistent=True) as tape:
      traj = imagine(policy, start, self.config.imag_horizon)
    metrics = self.update(traj, tape)
    return traj, metrics

  def update(self, traj, tape=None):
    tape = tape or tf.GradientTape()
    metrics = {}
    for key, critic in self.critics.items():
      mets = critic.train(traj, self.actor)
      metrics.update({f'{key}_{k}': v for k, v in mets.items()})
    with tape:
      scores = []
      for key, critic in self.critics.items():
        ret, baseline = critic.score(traj, self.actor)
        ret = self.retnorms[key](ret)
        baseline = self.retnorms[key](baseline, update=False)
        score = self.scorenorms[key](ret - baseline)
        metrics[f'{key}_score_mean'] = score.mean()
        metrics[f'{key}_score_std'] = score.std()
        metrics[f'{key}_score_mag'] = tf.abs(score).mean()
        metrics[f'{key}_score_max'] = tf.abs(score).max()
        scores.append(score * self.scales[key])
      score = self.advnorm(tf.reduce_sum(scores, 0))
      loss, mets = self.loss(traj, score)
      metrics.update(mets)
      loss = loss.mean()
    metrics.update(self.opt(tape, loss, self.actor))
    return metrics

  def loss(self, traj, score):
    metrics = {}
    policy = self.actor(tf.nest.map_structure(tf.stop_gradient, traj))
    action = tf.stop_gradient(traj['action'])
    if self.grad == 'backprop':
      loss = -score
    elif self.grad == 'reinforce':
      loss = -policy.log_prob(action)[:-1] * tf.stop_gradient(score)
    else:
      raise NotImplementedError(self.grad)

    shape = (
        self.act_space.shape[:-1] if self.act_space.discrete
        else self.act_space.shape)
    if self.config.actent_perdim and len(shape) > 0:
      assert isinstance(policy, tfd.Independent), type(policy)
      ent = policy.distribution.entropy()[:-1]
      if self.config.actent_norm:
        lo = policy.minent / ent.shape[-1]
        hi = policy.maxent / ent.shape[-1]
        ent = (ent - lo) / (hi - lo)
      ent_loss, mets = self.actent(ent)
      ent_loss = ent_loss.sum(-1)

    else:

      ent = policy.entropy()[:-1]
      if self.config.actent_norm:
        lo, hi = policy.minent, policy.maxent
        ent = (ent - lo) / (hi - lo)
      ent_loss, mets = self.actent(ent)

    metrics.update({f'actent_{k}': v for k, v in mets.items()})
    loss += ent_loss
    loss *= tf.stop_gradient(traj['weight'])[:-1]
    return loss, metrics


class MultiSkillImagActorCritic(tfutils.Module):

  def __init__(self, critics, scales, skill_shape, num_skills, config):
    critics = {k: v for k, v in critics.items() if scales[k]}
    for key, scale in scales.items():
      assert not scale or key in critics, key
    self.critics = {k: v for k, v in critics.items() if scales[k]}
    self.scales = scales
    self.num_skills = num_skills
    self.config = config

    self.act_space = {f'sk{i}': embodied.Space(
        np.int32 if 'onehot' in config.gpa.skill_vae.encoder.dist else np.float32, skill_shape) for i in range(num_skills)}
    self.act_space['choice'] = embodied.Space(np.int32, (num_skills,))
    self.choice_grads = config.gpa.manager.actor_choice_grad
    self.skill_grads = config.gpa.manager.actor_skill_grad
    head_updates = {f'sk{i}': {'dist': (config.actor_dist_disc if self.act_space[f'sk{i}'].discrete else config.actor_dist_cont)} for i in range(num_skills)}
    head_updates['choice'] = {'dist': 'onehot'}

    act_shape = {k: v.shape for k, v in self.act_space.items()}
    self.actor = nets.MLP(act_shape, **self.config.actor, head_updates=head_updates)

    self.advnorm = tfutils.Normalize(**self.config.advnorm)
    self.retnorms = {
        k: tfutils.Normalize(**self.config.retnorm) for k in self.critics}
    self.scorenorms = {
        k: tfutils.Normalize(**self.config.scorenorm) for k in self.critics}
    def get_actent(act_sp):
      if self.config.actent_perdim:
        shape = act_sp.shape[:-1] if act_sp.discrete else act_sp.shape
        return tfutils.AutoAdapt(
            shape, **self.config.actent, inverse=True)
      else:
        return tfutils.AutoAdapt((), **self.config.actent, inverse=True)
    self.actent = {k: get_actent(v) for k, v in self.act_space.items()}
    self.opt = tfutils.Optimizer('actor', **self.config.actor_opt)

  def initial(self, batch_size):
    return None

  def policy(self, state):
    outs = self.actor(state)
    skill_outs = tfutils.StackedDistributions([outs[f'sk{i}'] for i in range(self.num_skills)])
    outs['skill'] = skill_outs
    return outs

  def train(self, imagine, start, context):
    def policy(state):
      outs = self.policy(tf.nest.map_structure(tf.stop_gradient, state))
      return tf.nest.map_structure(lambda s: s.sample(), outs)
    with tf.GradientTape(persistent=True) as tape:
      traj = imagine(policy, start, self.config.imag_horizon)
    metrics = self.update(traj, tape)
    return traj, metrics

  def update(self, traj, tape=None):
    tape = tape or tf.GradientTape()
    metrics = {}
    for key, critic in self.critics.items():
      mets = critic.train(traj, self.actor)
      metrics.update({f'{key}_{k}': v for k, v in mets.items()})
    with tape:
      score, mets = self.score(traj)
      metrics.update(mets)
      goal_traj = {'deter': traj['goal_opts'],
                   'stoch': traj['goal_opts_stoch']}
      goal_value, mets = self.value(goal_traj)
      metrics.update(mets)
      loss, mets = self.loss(traj, score, goal_value)
      metrics.update(mets)
      loss = loss.mean()
    metrics.update(self.opt(tape, loss, self.actor))
    return metrics

  def score(self, traj):
    scores = []
    metrics = {}
    for key, critic in self.critics.items():
      ret, baseline = critic.score(traj, self.actor)
      ret = self.retnorms[key](ret)
      baseline = self.retnorms[key](baseline, update=False)
      score = self.scorenorms[key](ret - baseline)
      metrics[f'{key}_score_mean'] = score.mean()
      metrics[f'{key}_score_std'] = score.std()
      metrics[f'{key}_score_mag'] = tf.abs(score).mean()
      metrics[f'{key}_score_max'] = tf.abs(score).max()
      scores.append(score * self.scales[key])
    score = self.advnorm(tf.reduce_sum(scores, 0))
    return score, metrics

  def value(self, states):
    values = []
    metrics = {}
    for key, critic in self.critics.items():
      value = critic.target_net(states).mean()
      value = self.retnorms[key](value)
      metrics[f'{key}_value_mean'] = value.mean()
      metrics[f'{key}_value_std'] = value.std()
      values.append(value * self.scales[key])
    values = tf.reduce_sum(values, 0)
    return values, metrics

  def loss(self, traj, score, goal_value):
    metrics = {}
    traj = tf.nest.map_structure(tf.stop_gradient, traj)
    policy = self.policy(traj)
    metrics['manager_choice_hist'] = tf.argmax(policy['choice'].sample(), -1)
    rl_loss, mets = self.rl_loss(traj, policy, score, goal_value)
    losses = [rl_loss]
    metrics.update(mets)
    act_heads = set(self.act_space.keys())
    if not self.config.gpa.manager.choice_entropic_loss:
      act_heads.discard('choice')
    for head in act_heads:
      head_loss, mets = self.head_ent_loss(traj, policy, head)
      losses.append(head_loss)
      metrics.update({f'{head}_{k}': v for k, v in mets.items()})
    return tf.add_n(losses) * traj['weight'][:-1], metrics

  def rl_loss(self, traj, policy, score, goal_value):
    metrics = {}
    if self.skill_grads == 'backprop':
      skill_loss = -score
    elif self.skill_grads == 'reinforce':
      skill_loss = -policy['skill'].log_prob(traj['skill'])
      skill_loss = (skill_loss * traj['choice']).sum(-1)[:-1] * tf.stop_gradient(score)
    else:
      raise NotImplementedError(self.skill_grads)

    if self.choice_grads == 'backprop':
      choice_loss = -tf.reduce_sum(policy['choice'].sample() * tf.stop_gradient(goal_value), axis=-1)[:-1]
    elif self.choice_grads == 'reinforce':
      choice_loss = -policy['choice'].log_prob(traj['choice'])[:-1] * tf.stop_gradient(score)
    else:
      raise NotImplementedError(self.choice_grads)

    metrics['skill_loss'] = skill_loss.mean()
    metrics['choice_loss'] = choice_loss.mean()

    return (skill_loss + choice_loss), metrics

  def head_ent_loss(self, traj, policy, head):
    metrics = {}
    act_space = self.act_space[head]
    shape = (
      act_space.shape[:-1] if act_space.discrete
      else act_space.shape)
    if self.config.actent_perdim and len(shape) > 0:
      assert isinstance(policy[head], tfd.Independent), type(policy[head])
      ent = policy[head].distribution.entropy()[:-1]
      if self.config.actent_norm:
        lo = policy[head].minent / ent.shape[-1]
        hi = policy[head].maxent / ent.shape[-1]
        if not (hi == lo):
          ent = (ent - lo) / (hi - lo)
      ent_loss, mets = self.actent[head](ent)
      ent_loss = ent_loss.sum(-1)
    else:
      ent = policy[head].entropy()[:-1]
      if self.config.actent_norm:
        lo, hi = policy[head].minent, policy[head].maxent
        if not (hi == lo):
          ent = (ent - lo) / (hi - lo)
      ent_loss, mets = self.actent[head](ent)
    metrics.update({f'actent_{k}': v for k, v in mets.items()})
    return ent_loss, metrics


class VFunction(tfutils.Module):

  def __init__(self, rewfn, config):
    assert 'action' not in config.critic.inputs, config.critic.inputs
    self.rewfn = rewfn
    self.config = config
    self.net = nets.MLP((), **self.config.critic)
    if self.config.slow_target:
      self.target_net = nets.MLP((), **self.config.critic)
      self.updates = tf.Variable(-1, dtype=tf.int64)
    else:
      self.target_net = self.net
    self.opt = tfutils.Optimizer('critic', **self.config.critic_opt)

  def train(self, traj, actor):
    metrics = {}
    reward = self.rewfn(traj)
    target = tf.stop_gradient(self.target(
        traj, reward, self.config.critic_return)[0])
    with tf.GradientTape() as tape:
      dist = self.net({k: v[:-1] for k, v in traj.items()})
      loss = -(dist.log_prob(target) * traj['weight'][:-1]).mean()
    metrics.update(self.opt(tape, loss, self.net))
    metrics.update({
        'critic_loss': loss,
        'imag_reward_mean': reward.mean(),
        'imag_reward_std': reward.std(),
        'imag_critic_mean': dist.mean().mean(),
        'imag_critic_std': dist.mean().std(),
        'imag_return_mean': target.mean(),
        'imag_return_std': target.std(),
    })
    self.update_slow()
    return metrics

  def score(self, traj, actor):
    return self.target(traj, self.rewfn(traj), self.config.actor_return)

  def target(self, traj, reward, impl):
    if len(reward) != len(traj['action']) - 1:
      raise AssertionError('Should provide rewards for all but last action.')
    disc = traj['cont'][1:] * self.config.discount
    value = self.target_net(traj).mean()
    if impl == 'gae':
      advs = [tf.zeros_like(value[0])]
      deltas = reward + disc * value[1:] - value[:-1]
      for t in reversed(range(len(disc))):
        advs.append(deltas[t] + disc[t] * self.config.return_lambda * advs[-1])
      adv = tf.stack(list(reversed(advs))[:-1])
      return adv + value[:-1], value[:-1]
    elif impl == 'gve':
      vals = [value[-1]]
      interm = reward + disc * value[1:] * (1 - self.config.return_lambda)
      for t in reversed(range(len(disc))):
        vals.append(interm[t] + disc[t] * self.config.return_lambda * vals[-1])
      ret = tf.stack(list(reversed(vals))[:-1])
      return ret, value[:-1]
    else:
      raise NotImplementedError(impl)

  def update_slow(self):
    if not self.config.slow_target:
      return
    assert self.net.variables
    initialize = (self.updates == -1)
    if initialize or self.updates >= self.config.slow_target_update:
      self.updates.assign(0)
      mix = 1.0 if initialize else self.config.slow_target_fraction
      for s, d in zip(self.net.variables, self.target_net.variables):
        d.assign(mix * s + (1 - mix) * d)
    self.updates.assign_add(1)


class QFunction(tfutils.Module):

  def __init__(self, rewfn, config):
    assert config.actor_grad_disc == 'backprop'
    assert config.actor_grad_cont == 'backprop'
    assert 'action' in config.actor.inputs
    self.rewfn = rewfn
    self.config = config
    self.net = nets.MLP((), **self.config.critic)
    if self.config.slow_target:
      self.target_net = nets.MLP((), **self.config.critic)
      self.updates = tf.Variable(-1, dtype=tf.int64)
    else:
      self.target_net = self.net
    self.opt = tfutils.Optimizer('critic', **self.config.critic_opt)

  def score(self, traj, actor):
    traj = tf.nest.map_structure(tf.stop_gradient, traj)
    ret = self.net({**traj, 'action': actor(traj).sample()}).mode()[:-1]
    baseline = tf.zeros_like(ret)
    return ret, baseline

  def train(self, traj, actor):
    metrics = {}
    reward = self.rewfn(traj)
    target = tf.stop_gradient(self.target(traj, actor, reward))
    with tf.GradientTape() as tape:
      dist = self.net({k: v[:-1] for k, v in traj.items()})
      loss = -(dist.log_prob(target) * traj['weight'][:-1]).mean()
    metrics.update(self.opt(tape, loss, self.net))
    metrics.update({
        'imag_reward_mean': reward.mean(),
        'imag_reward_std': reward.std(),
        'imag_critic_mean': dist.mean().mean(),
        'imag_critic_std': dist.mean().std(),
        'imag_target_mean': target.mean(),
        'imag_target_std': target.std(),
    })
    self.update_slow()
    return metrics

  def target(self, traj, actor, reward):
    if len(reward) != len(traj['action']) - 1:
      raise AssertionError('Should provide rewards for all but last action.')
    cont = traj['cont'][1:]
    disc = cont * self.config.discount
    action = actor(traj).sample()
    value = self.target_net({**traj, 'action': action}).mean()
    if self.config.pengs_qlambda:
      vals = [value[-1]]
      interm = reward + disc * value[1:] * (1 - self.config.return_lambda)
      for t in reversed(range(len(disc))):
        vals.append(interm[t] + disc[t] * self.config.return_lambda * vals[-1])
      tar = tf.stack(list(reversed(vals))[:-1])
      return tar
    else:
      return reward + disc * value[1:]

  def update_slow(self):
    if not self.config.slow_target:
      return
    assert self.net.variables
    initialize = (self.updates == -1)
    if initialize or self.updates >= self.config.slow_target_update:
      self.updates.assign(0)
      mix = 1.0 if initialize else self.config.slow_target_fraction
      for s, d in zip(self.net.variables, self.target_net.variables):
        d.assign(mix * s + (1 - mix) * d)
    self.updates.assign_add(1)


class TwinQFunction(tfutils.Module):

  def __init__(self, rewfn, config):
    assert config.actor_grad_disc == 'backprop'
    assert config.actor_grad_cont == 'backprop'
    assert 'action' in config.actor.inputs
    self.rewfn = rewfn
    self.config = config
    self.net1 = nets.MLP((), **self.config.critic)
    self.net2 = nets.MLP((), **self.config.critic)
    if self.config.slow_target:
      self.target_net1 = nets.MLP((), **self.config.critic)
      self.target_net2 = nets.MLP((), **self.config.critic)
      self.updates = tf.Variable(-1, dtype=tf.int64)
    else:
      self.target_net1 = self.net1
      self.target_net2 = self.net2
    self.opt = tfutils.Optimizer('critic', **self.config.critic_opt)

  def score(self, traj, actor):
    traj = tf.nest.map_structure(tf.stop_gradient, traj)
    inps = {**traj, 'action': actor(traj).sample()}
    ret = tf.math.reduce_min([
        self.net1(inps).mode(),
        self.net2(inps).mode()], 0)[:-1]
    baseline = tf.zeros_like(ret)
    return ret, baseline

  def train(self, traj, actor):
    metrics = {}
    reward = self.rewfn(traj)
    target = tf.stop_gradient(self.target(traj, actor, reward))
    inps = {k: v[:-1] for k, v in traj.items()}
    with tf.GradientTape() as tape:
      dist1 = self.net1(inps)
      dist2 = self.net2(inps)
      loss1 = -(dist1.log_prob(target) * traj['weight'][:-1]).mean()
      loss2 = -(dist2.log_prob(target) * traj['weight'][:-1]).mean()
      loss = loss1 + loss2
    metrics.update(self.opt(tape, loss, [self.net1, self.net2]))
    metrics.update({
        'imag_reward_mean': reward.mean(),
        'imag_reward_std': reward.std(),
        'imag_critic_mean': dist1.mean().mean(),
        'imag_critic_std': dist2.mean().std(),
        'imag_target_mean': target.mean(),
        'imag_target_std': target.std(),
    })
    self.update_slow()
    return metrics

  def target(self, traj, actor, reward):
    if len(reward) != len(traj['action']) - 1:
      raise AssertionError('Should provide rewards for all but last action.')
    cont = traj['cont'][1:]
    disc = cont * self.config.discount
    value = tf.math.reduce_min([
        self.target_net1({**traj, 'action': actor(traj).sample()}).mean(),
        self.target_net2({**traj, 'action': actor(traj).sample()}).mean()], 0)
    if self.config.pengs_qlambda:
      vals = [value[-1]]
      interm = reward + disc * value[1:] * (1 - self.config.return_lambda)
      for t in reversed(range(len(disc))):
        vals.append(interm[t] + disc[t] * self.config.return_lambda * vals[-1])
      tar = tf.stack(list(reversed(vals))[:-1])
      return tar
    else:
      return reward + disc * value[1:]

  def update_slow(self):
    if not self.config.slow_target:
      return
    assert self.net1.variables
    assert self.net2.variables
    initialize = (self.updates == -1)
    if initialize or self.updates >= self.config.slow_target_update:
      self.updates.assign(0)
      mix = 1.0 if initialize else self.config.slow_target_fraction
      for s, d in zip(self.net1.variables, self.target_net2.variables):
        d.assign(mix * s + (1 - mix) * d)
      for s, d in zip(self.net2.variables, self.target_net2.variables):
        d.assign(mix * s + (1 - mix) * d)
    self.updates.assign_add(1)


class MultiSkillVAE(tfutils.Module):

  def __init__(self, state_shape, act_shape, config):
    self.state_shape = state_shape
    self.act_shape = act_shape
    self.skills = config.skills
    self.num_vaes = len(self.skills)
    self.config = config

    self.feat = nets.Input(['deter'])
    self.prior = tfutils.StackedDistributions([tfutils.get_prior(config.encoder, config.skill_shape)] * self.num_vaes)
    self.act_prior = tfutils.StackedDistributions([tfutils.get_prior(config.encoder.update({'dist': 'onehot'}), config.skill_shape)] * self.num_vaes) if config.encoder.dist == 'relaxed_onehot' else self.prior
    skill_shapes = {str(i): config.skill_shape for i in range(self.num_vaes)}
    self.enc = nets.MLP(
        skill_shapes, dims='context', **config.encoder)
    input_layer_cfg = config.decoder.update(
      {'dist': 'mse',
       'layers': 1})
    self.dec_input_layers = [nets.MLP(
        None, dims='context', **input_layer_cfg) for _ in range(self.num_vaes)]
    output_layer_cfg = config.decoder.update(
      {'inputs': ['tensor'],
       'layers': config.decoder.layers - 1})
    self.dec = nets.MLP(
        state_shape, **output_layer_cfg)
    self.kl_norms = [tfutils.AutoAdapt((), **config.norm_kl) for _ in range(self.num_vaes)]
    self.opt = tfutils.Optimizer('skill', **config.opt)

  def train(self, traj):
    metrics = {}
    feat = self.feat(traj)
    data = []
    for i in range(self.num_vaes):
      data.append(self.get_train_data(feat, traj, self.skills[i], metrics))
    with tf.GradientTape() as tape:
      loss, mets = self.loss(data)
    metrics.update(self.opt(tape, loss, [self.enc, self.dec] + self.dec_input_layers))
    metrics.update(mets)
    return metrics

  def loss(self, data):
    losses = []
    metrics = {}
    for i in range(self.num_vaes):
      context, goal, weight = data[i]
      enc = self.enc({'goal': goal, 'context': context})[str(i)]
      dec_input = self.dec_input_layers[i]({'skill': enc.sample(), 'context': context})
      dec = self.dec(dec_input)
      loss, mets = self.head_loss(enc, dec, context, goal, weight, i)
      losses.append(loss)
      metrics.update({f'sk{i}_{k}': v for k, v in mets.items()})
    total_loss = tf.add_n(losses)
    return total_loss, metrics

  def get_train_data(self, feat, traj, skill, metrics):
    if skill.startswith('T'):
      skill_length = int(skill[1:])
      assert feat.shape[0] >= skill_length, 'replay_length={} < skill_length={}'.format(feat.shape[0], skill_length)
      # feat = feat.reshape([feat.shape[0] // skill_length, skill_length] + feat.shape[1:])
      # context = feat[:, 0]
      # goal = feat[:, -1]
      context = feat[:-(skill_length - 1)]
      goal = feat[(skill_length - 1):]
      weight = tf.ones(context.shape[:-1])
    elif skill == 'span':
      context = feat[0]
      goal = feat[-1]
      weight = tf.ones(context.shape[:-1])
    elif skill == 'var':
      context = feat[:-self.config.train_skill_duration]
      final_goal = feat[-1]
      goal = tf.stack([final_goal] * context.shape[0], 0)
      weight = tf.ones(context.shape[:-1])
    else:
      raise NotImplementedError(skill)
    return tf.stop_gradient(context), tf.stop_gradient(goal), tf.stop_gradient(weight)

  def head_loss(self, enc, dec, context, goal, weight, index):
    metrics = {}
    rec = -dec.log_prob(tf.stop_gradient(goal))
    if self.config.kl_loss:
      kl = tfd.kl_divergence(enc, self.prior._dists[index])
      kl, mets = self.kl_norms[index](kl)
      if self.skills[index] == 'var':
        act_kl = tfd.kl_divergence(enc, self.act_prior._dists[index])
        best_skillength = context.shape[0] + self.config.train_skill_duration - tf.argmin(act_kl, 0)
        metrics['best_skillength_hist'] = best_skillength
      metrics.update({f'kl_{k}': v for k, v in mets.items()})
      assert rec.shape == kl.shape, (rec.shape, kl.shape)
    else:
      kl = 0.0
    skill_loss = ((rec + kl) * weight).mean()
    metrics['rec_mean'] = rec.mean()
    metrics['rec_std'] = rec.std()
    return skill_loss, metrics

  def encode(self, inputs):
    encs = self.enc(inputs)
    return tfutils.StackedDistributions([encs[str(i)] for i in range(self.num_vaes)])

  def decode(self, inputs: dict):
    skills = tf.unstack(inputs['skill'], num=self.num_vaes, axis=-(len(self.act_shape) + 1))
    dec_inputs = []
    for i in range(self.num_vaes):
      dec_input = self.dec_input_layers[i]({'context': inputs['context'], 'skill': skills[i]})
      dec_inputs.append(dec_input)
    dec_inputs = tf.stack(dec_inputs, axis=-(len(self.state_shape) + 1))
    dec = self.dec(dec_inputs)
    return dec

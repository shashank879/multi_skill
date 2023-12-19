# New
ipython3 embodied/agents/director/train.py -- \
    --logdir=logdir/dmc_cartpole_swingup/gpa_sk8x8_sl643216 \
    --configs=dmc_vision \
    --task=dmc_cartpole_swingup

# Debug
# ipython3 embodied/agents/director/train.py -- \
#     --logdir=logdir/debug/$(date +'%Y%m%d-%H%M%S') \
#     --configs=debug \
#     --task=dmc_walker_walk

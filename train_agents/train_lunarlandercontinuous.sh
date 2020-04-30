#!/bin/sh

MYDIR="$(dirname "$0")"
cd "$MYDIR"

multiply() {
    python -c "import numpy as np; args=[float(x) for x in '$*'.split()]; print(np.prod(args))"
}

multiply_int() {
    python -c "import numpy as np; args=[float(x) for x in '$*'.split()]; print(int(np.prod(args)))"
}

POPSIZE=200
POPSIZE_MAX=-1
MAX_SPEED=0.3
STDEV_LR=0.1

python train.py "$@" with \
    popsize="$POPSIZE" \
    popsize_max="$POPSIZE_MAX" \
    num_interactions=-1 \
    max_speed="$MAX_SPEED" \
    center_lr="$(multiply "$MAX_SPEED" 0.5)" \
    radius="$(multiply "$MAX_SPEED" 15)" \
    stdev_lr="$STDEV_LR" \
    obs_norm=False \
    hidden_size=0 \
    num_hidden=0 \
    decrease_rewards_by=0.0 \
    env_name="LunarLanderContinuous-v2" \
    re_eval_env_name="LunarLanderContinuous-v2" \
    niters=50

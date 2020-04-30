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
POPSIZE_MAX=3200
MAX_SPEED=0.015
STDEV_LR=0.1
RADIUS_MULTIPLIER=18

python train.py "$@" with \
    popsize="$POPSIZE" \
    popsize_max="$POPSIZE_MAX" \
    num_interactions="$(multiply_int "$POPSIZE" 0.75 1000)" \
    max_speed="$MAX_SPEED" \
    center_lr="$(multiply "$MAX_SPEED" 0.5)" \
    radius="$(multiply "$MAX_SPEED" "$RADIUS_MULTIPLIER")" \
    stdev_lr="$STDEV_LR" \
    obs_norm=True \
    hidden_size=0 \
    num_hidden=0 \
    decrease_rewards_by=5.0 \
    env_name="Humanoid-v2" \
    re_eval_env_name="Humanoid-v2" \
    niters=1000

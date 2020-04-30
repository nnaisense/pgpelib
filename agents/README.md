# Loading previously saved agents

This directory contains pickle files of previously saved agents.
Agents for the following environments are available:

- LunarLanderContinuous-v2 (box2d environment)
- Walker2d-v2 (MuJoCo environment)
- Humanoid-v2 (MuJoCo environment)
- HumanoidBulletEnv-v0 (pybullet environment)

All the agents, except for the `HumanoidBulletEnv-v0` agent, can be rendered
via:

```bash
python enjoy.py <SAVED_AGENT_FILE_NAME>.pickle
```

The `HumanoidBulletEnv-v0` is a special case here because it was trained
according to a manually defined gym environment (in which the survival bonus
is removed from the reward function and the episode length is shortened from
1000 to 200). However, for visualization, we would like to demonstrate how the
agent performs according to the standard reward function, with full episode
length. For this reason, a different visualization script specific to the
`HumanoidBulletEnv-v0` agent is available and can be executed via:

```bash
python enjoy_pybullet_humanoid.py <SAVED_AGENT_FILE_NAME>.pickle
```

## Requirements

For visualizing the LunarLanderContinuous agent, `gym[box2d]` needs to be installed.
For the MuJoCo agents, `gym[mujoco]` is required.
For the HumanoidBulletEnv agent, `pybullet` is required.
All these requirements can be satisfied using the following pip installation command:

```bash
pip install gym[box2d,mujoco] pybullet
```

# Training reinforcement learning agents using pgpelib

This directory contains a Python script for training reinforcement learning (RL) agents.
The training script uses the [sacred](https://github.com/IDSIA/sacred) library for reporting and storing the evolution progress.

All the configurations (like, e.g., `'env_name'`, the name of the [gym](https://gym.openai.com/) environment to solve) and hyperparameters (like, e.g., `'popsize'`, the population size; `'center_lr'`, the learning rate for updating the center solution, etc.) can be seen by typing:

```bash
python train.py print_config
```

By default, the script is tuned to solve `HumanoidBulletEnv-v0`, the humanoid task of the [PyBullet library](https://pybullet.org).
Simply running the script will therefore run a new `HumanoidBulletEnv-v0` experiment (and will reproduce our reported PyBullet Humanoid results from the ClipUp paper):

```bash
python train.py
```

However, instead of simply running the script like above, you will most probably want to pass certain command line options to save the logs of the experiment somewhere.
One option is to save the results in a directory (say, `results/`), by running the experiment like this:

```bash
python train.py -F results
```

Another option is to save the results into a mongodb database by running the experiment like this:

```bash
python train.py -m IP:PORT:DATABASE_NAME
```

There are also shell scripts which run `train.py` for solving MuJoCo `Humanoid-v2` and `Walker2d-v2` tasks. The usefulness of these shell scripts are two-fold:

- These scripts serve as a demonstrate how to run `train.py` and how to specify the target environment name and hyperparameter values.
- These scripts allow one to reproduce the PGPE+ClipUp results on MuJoCo locomotion tasks reported in our ClipUp paper.

For example, for analyzing the hyperparameters used for solving the `Walker2d-v2` task, one can run the following shell command:

```bash
./train_walker2d.sh print_config
```

For solving the `Walker2d-v2` task, one can run the following shell command:

```bash
./train_walker2d.sh -F results
```

or:

```bash
./train_walker2d.sh -m IP:PORT:DATABASE_NAME
```

## How to load saved agents

The experiment script periodically creates pickle files in which the parameters and further metadata regarding the current policy are stored.
Those saved policies can be restored as [PyTorch](https://pytorch.org/) modules as shown below:

```python
from pgpelib.restore import to_torch_module

policy_module = to_torch_module('filename.pickle')
```

To re-create the [gym](https://gym.openai.com/) environment for which the policy was trained, one can do the following:

```python
from pgpelib.restore import policy_metadata
import gym

metadata = policy_metadata('filename.pickle')
env_name = metadata['env_name']
env_config = metadata['env_config']
env = gym.make(env_name, **env_config)
```


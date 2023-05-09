# Dual Curriculum Design with Mujoco Support

[![License: CC BY-SA 4.0](https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-sa/4.0/)

### RoML

[RoML](https://github.com/fictivename/RoML-varibad) (Robust Meta Reinforcement Learning) is a meta reinforcement learning method. Instead of the average return of the agent, it optimizes a risk measure, which assigns more weight to high-risk tasks. Specifically, it optimizes the Conditional Value at Risk (CVaR), i.e., the average over the worst alpha quantile of the tasks.

### This repo: UED as a baseline for comparison

As a reference for comparison with RoML, **this repository implements Unsupervised Environment Design (UED)**. UED is an approach for robust reinforcement learning, which minimizes the agent regret wrt alternative agents.
We rely on the [official DCD repository](https://github.com/facebookresearch/dcd), and extend the UED's algorithms [PAIRED](https://arxiv.org/abs/2012.02096), [REPAIRED](https://arxiv.org/abs/2110.02439), [ACCEL](https://accelagent.github.io/) and [Robust PLR](https://arxiv.org/abs/2110.02439) to support the same mujoco environments on which ROML was tested.
For each algorithm, we use the hyper-parameters from the [bipedal environment](https://github.com/facebookresearch/dcd/tree/main/train_scripts/grid_configs/bipedal), as defined in the original DCD repository.
Note that empirically, only PAIRED managed to produce meaningful learning in the mujoco environments.

### How to run

Example - to train and test PAIRED on HalfCheetahMass:

`main.py --num_env_steps=100000000 --env_name=HalfCheetahMassAdversarial-v0 --seed=88 --ued_algo=paired --use_plr=False --use_editor=False`

For another environment, modify `env_name`.
For another algorithm:
* PAIRED: `--ued_algo=paired --use_plr=False --use_editor=False` (as above)
* REPAIRED: `--ued_algo=paired --use_plr=True --use_editor=False`
* RobustPLR: `--ued_algo=domain_randomization --use_plr=True --use_editor=False`
* ACCEL: `--ued_algo=domain_randomization --use_plr=True --use_editor=True`

### Credits

This repository is based on the [official DCD repository](https://github.com/facebookresearch/dcd).
See the original repo for more details about the framework and implementation.
Our main changes are the additional HalfCheetah environments under `envs/`, as well as `main.py` to merge training and evaluation.

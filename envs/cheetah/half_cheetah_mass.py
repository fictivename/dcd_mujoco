import numpy as np
import random
from gym.spaces import Box

from .half_cheetah import HalfCheetahEnv

from . import register


class HalfCheetahMassEnv(HalfCheetahEnv):
    """Half-cheetah environment with target velocity, as described in [1]. The
    code is adapted from
    https://github.com/cbfinn/maml_rl/blob/9c8e2ebd741cb0c7b8bf2d040c4caeeb8e06cc95/rllab/envs/mujoco/half_cheetah_env_rand.py
    The half-cheetah follows the dynamics from MuJoCo [2], and receives at each
    time step a reward composed of a control cost and a penalty equal to the
    difference between its current velocity and the target velocity. The tasks
    are generated by sampling the target velocities from the uniform
    distribution on [0, 2].
    [1] Chelsea Finn, Pieter Abbeel, Sergey Levine, "Model-Agnostic
        Meta-Learning for Fast Adaptation of Deep Networks", 2017
        (https://arxiv.org/abs/1703.03400)
    [2] Emanuel Todorov, Tom Erez, Yuval Tassa, "MuJoCo: A physics engine for
        model-based control", 2012
        (https://homes.cs.washington.edu/~todorov/papers/TodorovIROS12.pdf)
    """

    def __init__(self, task=1, n_tasks=2, randomize_tasks=True, seed=0):
        self._task = task
        # self._idx = 0
        self._max_episode_steps = 200
        self._time = 0
        self._return = 0
        self._last_return = 0
        self._curr_rets = []

        super(HalfCheetahMassEnv, self).__init__()

        self.original_mass_vec = self.model.body_mass.copy()  # 8 elements
        self.set_task(self.sample_task())

        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(20,), dtype=np.float64)

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]

        forward_reward = (xposafter - xposbefore) / self.dt
        ctrl_cost = 0.5 * 1e-1 * np.sum(np.square(action))

        observation = self._get_obs()

        # this value is in [-0.1,0.1] when the cheetah is straight, and in [-0.6,-0.4] when it's upside-down.
        #  we penalize the cheetah being upside down.
        reward_height = observation[0]
        reward = forward_reward - ctrl_cost + reward_height

        done = False
        infos = dict(reward_forward=forward_reward,
                     reward_ctrl=-ctrl_cost, task=self._task)

        self._time += 1
        self._return += reward
        if self._time % self._max_episode_steps == 0:
            # print(f'[{self._time//self._max_episode_steps}] '
            #       f'{self.task},\t{self._return}')
            self._last_return = self._return
            self._curr_rets.append(self._return)
            self._return = 0
        return (observation, reward, done, infos)

    def get_task(self):
        return self._task

    def sample_task(self):
        return 2 ** random.uniform(-1, 1)
        # return 1 ** random.uniform(-1, 1)  # TODO sanity

    @property
    def processed_action_dim(self):
        return 1

    def sample_tasks(self, n_tasks):
        return [self.sample_task() for _ in range(n_tasks)]
    #
    # def sample_tasks(self, num_tasks):
    #     # np.random.seed(1337)
    #     factors = 2 ** np.random.uniform(-1, 1, size=(num_tasks,))
    #     tasks = [{'mass_factor': fac} for fac in factors]
    #     return tasks
    #
    # def sample_task(self):
    #     return self.sample_tasks(1)

    def set_task(self, task):
        if isinstance(task, np.ndarray):
            task = task[0]
        self._task = task
        for i in range(len(self.model.body_mass)):
            self.model.body_mass[i] = task * self.original_mass_vec[i]
        self.model.geom_size[1:, 0] = 0.046 * task
        return task

    # def get_all_task_idx(self):
    #     return range(len(self.tasks))

    def get_last_return(self):
        return np.sum(self._curr_rets)

    # def get_task_return(self, idx=None):
    #     if idx is None:
    #         idx = self._idx
    #     return self._curr_return[idx]

    def reset_task(self, task):
        if task is None:
            task = self.sample_task()
        self.set_task(task)
        self._time = 0
        self._last_return = self._return
        self._curr_rets = []
        self._return = 0
        # self.reset()

    # def reset_task(self, idx, task=None, resample_task=False):
    #     self._idx = idx
    #     if task is not None:
    #         if not isinstance(task, dict):
    #             task = dict(mass_factor=task)
    #         self.tasks[idx] = task
    #     elif resample_task:
    #         self.tasks[idx] = self.sample_tasks(1)[0]
    #     self._task = self.tasks[idx]
    #     self._curr_steps[self._idx] = 0
    #     self._curr_return[self._idx] = 0
    #     self.set_task()
    #     self.reset()


if hasattr(__loader__, 'name'):
  module_path = __loader__.name
elif hasattr(__loader__, 'fullname'):
  module_path = __loader__.fullname

register.register(
    env_id='HalfCheetahMass-v0',
    entry_point=module_path + ':HalfCheetahMassEnv',
    max_episode_steps=200,
)

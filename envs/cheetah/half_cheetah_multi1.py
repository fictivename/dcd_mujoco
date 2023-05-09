import numpy as np
import random
from gym.spaces import Box

from .half_cheetah import HalfCheetahEnv

from . import register

MODEL_KEYS = ['actuator_gear', 'actuator_lengthrange', 'body_inertia', 'cam_ipd', 'geom_solmix',
              'geom_solref', 'jnt_margin', 'jnt_stiffness', 'light_pos', 'mat_rgba']

class HalfCheetahMulti1Env(HalfCheetahEnv):
    def __init__(self, task=(1,1,1,1,1,1,1,1,1,1), n_tasks=2, randomize_tasks=True, seed=0):
        self.task_dim = 10
        self._task = task
        # self._idx = 0
        self._max_episode_steps = 200
        self._time = 0
        self._return = 0
        self._last_return = 0
        self._curr_rets = []

        super(HalfCheetahMulti1Env, self).__init__()

        self.model_keys = MODEL_KEYS
        self.original_vecs = [getattr(self.model, k).copy()
                              for k in self.model_keys]

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
        return np.array([2 ** random.uniform(-0.5, 0.5)
                         for _ in range(self.task_dim)])

    @property
    def processed_action_dim(self):
        return 10

    def sample_tasks(self, n_tasks):
        return [self.sample_task() for _ in range(n_tasks)]

    def set_task(self, task):
        # if isinstance(task, np.ndarray):
        #     task = task[0]
        self._task = task

        for i, k in enumerate(self.model_keys):
            for j in range(len(self.original_vecs[i])):
                getattr(self.model, k)[j] = task[i] * self.original_vecs[i][j]

        return task

    def get_last_return(self):
        return np.sum(self._curr_rets)

    def reset_task(self, task):
        if task is None:
            task = self.sample_task()
        self.set_task(task)
        self._time = 0
        self._last_return = self._return
        self._curr_rets = []
        self._return = 0
        # self.reset()


if hasattr(__loader__, 'name'):
  module_path = __loader__.name
elif hasattr(__loader__, 'fullname'):
  module_path = __loader__.fullname

register.register(
    env_id='HalfCheetahMulti1-v0',
    entry_point=module_path + ':HalfCheetahMulti1Env',
    max_episode_steps=200,
)

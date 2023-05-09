import random

import os
import numpy as np
import torch

import gym
from gym import spaces
from .half_cheetah import HalfCheetahEnv
from .half_cheetah_vel import HalfCheetahVelEnv
from .half_cheetah_mass import HalfCheetahMassEnv
from .half_cheetah_body import HalfCheetahBodyEnv
from .half_cheetah_multi1 import HalfCheetahMulti1Env
from .half_cheetah_multi2 import HalfCheetahMulti2Env
from .half_cheetah_multi3 import HalfCheetahMulti3Env

from . import register

class HalfCheetahVelAdversarialEnv(HalfCheetahVelEnv):
    def __init__(self, max_episode_steps=200, seed=0):
        self._max_episode_steps = max_episode_steps
        self.task_dim = 1
        super(HalfCheetahVelAdversarialEnv, self).__init__(seed=seed)

        self.adversary_action_space = spaces.Box(
            low=-1, high=1, shape=(self.task_dim,), dtype=np.float32)

        # self.adversary_observation_space = spaces.Box(
        #     low=-1, high=1, shape=(0,), dtype=np.float32)
        self.adversary_image_obs_space = spaces.Box(
            low=-1, high=1, shape=(self.task_dim,), dtype=np.float32)
        self.adversary_ts_obs_space = gym.spaces.Box(
            low=0, high=2, shape=(1,), dtype='uint8')
        self.adversary_randomz_obs_space = gym.spaces.Box(
            low=0, high=1.0, shape=(50,), dtype=np.float32)
        self.adversary_observation_space = gym.spaces.Dict(
            {'image': self.adversary_image_obs_space,
             'time_step': self.adversary_ts_obs_space,
             'random_z': self.adversary_randomz_obs_space})

        self.set_task(self.sample_task())

        self.level_seed = 0
        self.encoding = np.array([self._task] + [self.level_seed])
        self.passable = True

        self._time = 0
        self._return = 0
        self._last_return = 0
        self._curr_rets = []

    def reset_agent(self):
        return super().reset()

    def reset_random(self):
        self.set_task(self.sample_task())
        return super().reset()

    def reset(self):
        self._time = 0
        # obs0 = super().reset()
        obs = {
            # 'image': obs0,
            'image': [1],
            'time_step': [self._time],
            'random_z': np.random.uniform(size=(50,)).astype(np.float32),
        }
        return obs

    def step_adversary(self, action):
        if torch.is_tensor(action):
            action = action.item()
        self.set_task(7*(action+1)/2)
        done = False
        obs = {
            'image': np.array([action]),
            'time_step': [self._time],
            'random_z': np.random.uniform(size=(50,)).astype(np.float32)
        }
        return obs, 0, done, {}

    @property
    def level(self):
        return tuple(self.encoding)

    def reset_to_level(self, level):
        self.reset()

        if isinstance(level, str):
            encoding = list(np.fromstring(level))
        else:
            encoding = [float(x) for x in level[:-1]] + [int(level[-1])]

        assert len(level) == self.task_dim + 1, \
            f'Level input is the wrong length.'

        self.set_task(encoding[:-1])
        self.level_seed = encoding[-1]
        self.encoding = np.array(encoding)

        return self.reset_agent()

    def mutate_level(self, num_edits=1):
        for _ in range(num_edits):
            i_mutation = np.random.randint(self.task_dim)
            task = self._task
            task[i_mutation] = np.random.uniform(0, 7)
            self.set_task(task)
            self.level_seed = rand_int_seed()
        return self.reset_agent()


class HalfCheetahMassAdversarialEnv(HalfCheetahMassEnv):
    def __init__(self, max_episode_steps=200, seed=0):
        self._max_episode_steps = max_episode_steps
        self.task_dim = 1
        super(HalfCheetahMassAdversarialEnv, self).__init__(seed=seed)

        self.adversary_action_space = spaces.Box(
            low=-1, high=1, shape=(self.task_dim,), dtype=np.float32)

        # self.adversary_observation_space = spaces.Box(
        #     low=-1, high=1, shape=(0,), dtype=np.float32)
        self.adversary_image_obs_space = spaces.Box(
            low=-1, high=1, shape=(self.task_dim,), dtype=np.float32)
        self.adversary_ts_obs_space = gym.spaces.Box(
            low=0, high=2, shape=(1,), dtype='uint8')
        self.adversary_randomz_obs_space = gym.spaces.Box(
            low=0, high=1.0, shape=(50,), dtype=np.float32)
        self.adversary_observation_space = gym.spaces.Dict(
            {'image': self.adversary_image_obs_space,
             'time_step': self.adversary_ts_obs_space,
             'random_z': self.adversary_randomz_obs_space})

        self.original_mass_vec = self.model.body_mass.copy()  # 8 elements
        self.set_task(self.sample_task())

        self.level_seed = 0
        self.encoding = np.array([self._task] + [self.level_seed])
        self.passable = True

        self._time = 0
        self._return = 0
        self._last_return = 0
        self._curr_rets = []

    def reset_agent(self):
        return super().reset()

    def reset_random(self):
        self.set_task(self.sample_task())
        return super().reset()

    def reset(self):
        self._time = 0
        # obs0 = super().reset()
        obs = {
            # 'image': obs0,
            'image': [1],
            'time_step': [self._time],
            'random_z': np.random.uniform(size=(50,)).astype(np.float32),
        }
        return obs

    def step_adversary(self, action):
        if torch.is_tensor(action):
            action = action.item()
        self.set_task(2.0**action)
        # self.set_task(1.0**action)  # TODO sanity
        done = False
        # obs = np.array([])
        obs = {
            'image': np.array([action]),
            'time_step': [self._time],
            'random_z': np.random.uniform(size=(50,)).astype(np.float32)
        }
        return obs, 0, done, {}

    @property
    def level(self):
        return tuple(self.encoding)

    def reset_to_level(self, level):
        self.reset()

        if isinstance(level, str):
            encoding = list(np.fromstring(level))
        else:
            encoding = [float(x) for x in level[:-1]] + [int(level[-1])]

        assert len(level) == self.task_dim + 1, \
            f'Level input is the wrong length.'

        self.set_task(encoding[:-1])
        self.level_seed = encoding[-1]
        self.encoding = np.array(encoding)

        return self.reset_agent()

    def mutate_level(self, num_edits=1):
        for _ in range(num_edits):
            i_mutation = np.random.randint(self.task_dim)
            task = self._task
            task[i_mutation] = 2 ** np.random.uniform(-1,1)
            self.set_task(task)
            self.level_seed = rand_int_seed()
        return self.reset_agent()


class HalfCheetahBodyAdversarialEnv(HalfCheetahBodyEnv):
    def __init__(self, max_episode_steps=200, seed=0):
        self._max_episode_steps = max_episode_steps
        super(HalfCheetahBodyAdversarialEnv, self).__init__(seed=seed)

        self.adversary_action_space = spaces.Box(
            low=-1, high=1, shape=(self.task_dim,), dtype=np.float32)

        # self.adversary_observation_space = spaces.Box(
        #     low=-1, high=1, shape=(0,), dtype=np.float32)
        self.adversary_image_obs_space = spaces.Box(
            low=-1, high=1, shape=(self.task_dim,), dtype=np.float32)
        self.adversary_ts_obs_space = gym.spaces.Box(
            low=0, high=2, shape=(1,), dtype='uint8')
        self.adversary_randomz_obs_space = gym.spaces.Box(
            low=0, high=1.0, shape=(50,), dtype=np.float32)
        self.adversary_observation_space = gym.spaces.Dict(
            {'image': self.adversary_image_obs_space,
             'time_step': self.adversary_ts_obs_space,
             'random_z': self.adversary_randomz_obs_space})

        self.set_task(self.sample_task())

        self.level_seed = 0
        self.encoding = np.array(list(self._task) + [self.level_seed])
        self.passable = True

        self._time = 0
        self._return = 0
        self._last_return = 0
        self._curr_rets = []

    def reset_agent(self):
        return super().reset()

    def reset_random(self):
        self.set_task(self.sample_task())
        return super().reset()

    def reset(self):
        self._time = 0
        # obs0 = super().reset()
        obs = {
            # 'image': obs0,
            'image': [1],
            'time_step': [self._time],
            'random_z': np.random.uniform(size=(50,)).astype(np.float32),
        }
        return obs

    def step_adversary(self, action):
        if torch.is_tensor(action):
            action = action.detach().cpu().numpy()
        self.set_task(2.0**action)
        done = False
        # obs = np.array([])
        obs = {
            'image': np.array([action]),
            'time_step': [self._time],
            'random_z': np.random.uniform(size=(50,)).astype(np.float32)
        }
        return obs, 0, done, {}

    @property
    def level(self):
        return tuple(self.encoding)

    def reset_to_level(self, level):
        self.reset()

        if isinstance(level, str):
            encoding = list(np.fromstring(level))
        else:
            encoding = [float(x) for x in level[:-1]] + [int(level[-1])]

        assert len(level) == self.task_dim + 1, \
            f'Level input is the wrong length.'

        self.set_task(encoding[:-1])
        self.level_seed = encoding[-1]
        self.encoding = np.array(encoding)

        return self.reset_agent()

    def mutate_level(self, num_edits=1):
        for _ in range(num_edits):
            i_mutation = np.random.randint(self.task_dim)
            task = self._task
            task[i_mutation] = 2 ** np.random.uniform(-1,1)
            self.set_task(task)
            self.level_seed = rand_int_seed()
        return self.reset_agent()


class HalfCheetahMulti1AdversarialEnv(HalfCheetahBodyEnv):
    def __init__(self, max_episode_steps=200, seed=0):
        self._max_episode_steps = max_episode_steps
        super(HalfCheetahMulti1AdversarialEnv, self).__init__(seed=seed)

        self.adversary_action_space = spaces.Box(
            low=-1, high=1, shape=(self.task_dim,), dtype=np.float32)

        # self.adversary_observation_space = spaces.Box(
        #     low=-1, high=1, shape=(0,), dtype=np.float32)
        self.adversary_image_obs_space = spaces.Box(
            low=-1, high=1, shape=(self.task_dim,), dtype=np.float32)
        self.adversary_ts_obs_space = gym.spaces.Box(
            low=0, high=2, shape=(1,), dtype='uint8')
        self.adversary_randomz_obs_space = gym.spaces.Box(
            low=0, high=1.0, shape=(50,), dtype=np.float32)
        self.adversary_observation_space = gym.spaces.Dict(
            {'image': self.adversary_image_obs_space,
             'time_step': self.adversary_ts_obs_space,
             'random_z': self.adversary_randomz_obs_space})

        self.set_task(self.sample_task())

        self.level_seed = 0
        self.encoding = np.array(list(self._task) + [self.level_seed])
        self.passable = True

        self._time = 0
        self._return = 0
        self._last_return = 0
        self._curr_rets = []

    def reset_agent(self):
        return super().reset()

    def reset_random(self):
        self.set_task(self.sample_task())
        return super().reset()

    def reset(self):
        self._time = 0
        # obs0 = super().reset()
        obs = {
            # 'image': obs0,
            'image': [1],
            'time_step': [self._time],
            'random_z': np.random.uniform(size=(50,)).astype(np.float32),
        }
        return obs

    def step_adversary(self, action):
        if torch.is_tensor(action):
            action = action.detach().cpu().numpy()
        self.set_task(2.0**(0.5*action))
        done = False
        # obs = np.array([])
        obs = {
            'image': np.array([action]),
            'time_step': [self._time],
            'random_z': np.random.uniform(size=(50,)).astype(np.float32)
        }
        return obs, 0, done, {}

    @property
    def level(self):
        return tuple(self.encoding)

    def reset_to_level(self, level):
        self.reset()

        if isinstance(level, str):
            encoding = list(np.fromstring(level))
        else:
            encoding = [float(x) for x in level[:-1]] + [int(level[-1])]

        assert len(level) == self.task_dim + 1, \
            f'Level input is the wrong length.'

        self.set_task(encoding[:-1])
        self.level_seed = encoding[-1]
        self.encoding = np.array(encoding)

        return self.reset_agent()

    def mutate_level(self, num_edits=1):
        for _ in range(num_edits):
            i_mutation = np.random.randint(self.task_dim)
            task = self._task
            task[i_mutation] = 2 ** np.random.uniform(-1,1)
            self.set_task(task)
            self.level_seed = rand_int_seed()
        return self.reset_agent()

class HalfCheetahMulti2AdversarialEnv(HalfCheetahBodyEnv):
    def __init__(self, max_episode_steps=200, seed=0):
        self._max_episode_steps = max_episode_steps
        super(HalfCheetahMulti2AdversarialEnv, self).__init__(seed=seed)

        self.adversary_action_space = spaces.Box(
            low=-1, high=1, shape=(self.task_dim,), dtype=np.float32)

        # self.adversary_observation_space = spaces.Box(
        #     low=-1, high=1, shape=(0,), dtype=np.float32)
        self.adversary_image_obs_space = spaces.Box(
            low=-1, high=1, shape=(self.task_dim,), dtype=np.float32)
        self.adversary_ts_obs_space = gym.spaces.Box(
            low=0, high=2, shape=(1,), dtype='uint8')
        self.adversary_randomz_obs_space = gym.spaces.Box(
            low=0, high=1.0, shape=(50,), dtype=np.float32)
        self.adversary_observation_space = gym.spaces.Dict(
            {'image': self.adversary_image_obs_space,
             'time_step': self.adversary_ts_obs_space,
             'random_z': self.adversary_randomz_obs_space})

        self.set_task(self.sample_task())

        self.level_seed = 0
        self.encoding = np.array(list(self._task) + [self.level_seed])
        self.passable = True

        self._time = 0
        self._return = 0
        self._last_return = 0
        self._curr_rets = []

    def reset_agent(self):
        return super().reset()

    def reset_random(self):
        self.set_task(self.sample_task())
        return super().reset()

    def reset(self):
        self._time = 0
        # obs0 = super().reset()
        obs = {
            # 'image': obs0,
            'image': [1],
            'time_step': [self._time],
            'random_z': np.random.uniform(size=(50,)).astype(np.float32),
        }
        return obs

    def step_adversary(self, action):
        if torch.is_tensor(action):
            action = action.detach().cpu().numpy()
        self.set_task(2.0**(0.5*action))
        done = False
        # obs = np.array([])
        obs = {
            'image': np.array([action]),
            'time_step': [self._time],
            'random_z': np.random.uniform(size=(50,)).astype(np.float32)
        }
        return obs, 0, done, {}

    @property
    def level(self):
        return tuple(self.encoding)

    def reset_to_level(self, level):
        self.reset()

        if isinstance(level, str):
            encoding = list(np.fromstring(level))
        else:
            encoding = [float(x) for x in level[:-1]] + [int(level[-1])]

        assert len(level) == self.task_dim + 1, \
            f'Level input is the wrong length.'

        self.set_task(encoding[:-1])
        self.level_seed = encoding[-1]
        self.encoding = np.array(encoding)

        return self.reset_agent()

    def mutate_level(self, num_edits=1):
        for _ in range(num_edits):
            i_mutation = np.random.randint(self.task_dim)
            task = self._task
            task[i_mutation] = 2 ** np.random.uniform(-1,1)
            self.set_task(task)
            self.level_seed = rand_int_seed()
        return self.reset_agent()

class HalfCheetahMulti3AdversarialEnv(HalfCheetahBodyEnv):
    def __init__(self, max_episode_steps=200, seed=0):
        self._max_episode_steps = max_episode_steps
        super(HalfCheetahMulti3AdversarialEnv, self).__init__(seed=seed)

        self.adversary_action_space = spaces.Box(
            low=-1, high=1, shape=(self.task_dim,), dtype=np.float32)

        # self.adversary_observation_space = spaces.Box(
        #     low=-1, high=1, shape=(0,), dtype=np.float32)
        self.adversary_image_obs_space = spaces.Box(
            low=-1, high=1, shape=(self.task_dim,), dtype=np.float32)
        self.adversary_ts_obs_space = gym.spaces.Box(
            low=0, high=2, shape=(1,), dtype='uint8')
        self.adversary_randomz_obs_space = gym.spaces.Box(
            low=0, high=1.0, shape=(50,), dtype=np.float32)
        self.adversary_observation_space = gym.spaces.Dict(
            {'image': self.adversary_image_obs_space,
             'time_step': self.adversary_ts_obs_space,
             'random_z': self.adversary_randomz_obs_space})

        self.set_task(self.sample_task())

        self.level_seed = 0
        self.encoding = np.array(list(self._task) + [self.level_seed])
        self.passable = True

        self._time = 0
        self._return = 0
        self._last_return = 0
        self._curr_rets = []

    def reset_agent(self):
        return super().reset()

    def reset_random(self):
        self.set_task(self.sample_task())
        return super().reset()

    def reset(self):
        self._time = 0
        # obs0 = super().reset()
        obs = {
            # 'image': obs0,
            'image': [1],
            'time_step': [self._time],
            'random_z': np.random.uniform(size=(50,)).astype(np.float32),
        }
        return obs

    def step_adversary(self, action):
        if torch.is_tensor(action):
            action = action.detach().cpu().numpy()
        self.set_task(2.0**(0.5*action))
        done = False
        # obs = np.array([])
        obs = {
            'image': np.array([action]),
            'time_step': [self._time],
            'random_z': np.random.uniform(size=(50,)).astype(np.float32)
        }
        return obs, 0, done, {}

    @property
    def level(self):
        return tuple(self.encoding)

    def reset_to_level(self, level):
        self.reset()

        if isinstance(level, str):
            encoding = list(np.fromstring(level))
        else:
            encoding = [float(x) for x in level[:-1]] + [int(level[-1])]

        assert len(level) == self.task_dim + 1, \
            f'Level input is the wrong length.'

        self.set_task(encoding[:-1])
        self.level_seed = encoding[-1]
        self.encoding = np.array(encoding)

        return self.reset_agent()

    def mutate_level(self, num_edits=1):
        for _ in range(num_edits):
            i_mutation = np.random.randint(self.task_dim)
            task = self._task
            task[i_mutation] = 2 ** np.random.uniform(-1,1)
            self.set_task(task)
            self.level_seed = rand_int_seed()
        return self.reset_agent()


def rand_int_seed():
    return int.from_bytes(os.urandom(4), byteorder="little")

if hasattr(__loader__, 'name'):
  module_path = __loader__.name
elif hasattr(__loader__, 'fullname'):
  module_path = __loader__.fullname

register.register(
    env_id='HalfCheetahVelAdversarial-v0',
    entry_point=module_path + ':HalfCheetahVelAdversarialEnv',
    max_episode_steps=200,
)

register.register(
    env_id='HalfCheetahMassAdversarial-v0',
    entry_point=module_path + ':HalfCheetahMassAdversarialEnv',
    max_episode_steps=200,
)

register.register(
    env_id='HalfCheetahBodyAdversarial-v0',
    entry_point=module_path + ':HalfCheetahBodyAdversarialEnv',
    max_episode_steps=200,
)

register.register(
    env_id='HalfCheetahMulti1Adversarial-v0',
    entry_point=module_path + ':HalfCheetahMulti1AdversarialEnv',
    max_episode_steps=200,
)

register.register(
    env_id='HalfCheetahMulti2Adversarial-v0',
    entry_point=module_path + ':HalfCheetahMulti2AdversarialEnv',
    max_episode_steps=200,
)

register.register(
    env_id='HalfCheetahMulti3Adversarial-v0',
    entry_point=module_path + ':HalfCheetahMulti3AdversarialEnv',
    max_episode_steps=200,
)

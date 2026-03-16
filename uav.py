
import gym
import numpy as np
from matplotlib import pyplot as plt

from arg_data import CarsPath
from channel import get_fso_capacity, calculate_optimal_divergence
from store_file import Buffer


uav_height = 100  # m
# target_rate = 4.0e2  # Mbs
slot_time = 1  # s
fso_power = 15  # dBm
theta_max = np.pi / 2
omega = 5

# Fixed UAV position
uav_pos_fixed = np.array([220, 220, uav_height], dtype=np.float32)

class MakeEnv(gym.Env):
    def __init__(self, set_num, target_rate):
        self.car_num = set_num
        # load
        self.cars_path = CarsPath()
        self._max_episode_steps = self.cars_path.max_time
        # store
        self.buffer = Buffer(max_time=self._max_episode_steps + 1, car_num=self.car_num)
        self.p_fso_max = fso_power
        # edge constraint
        self.target_rate = target_rate
        self.delta_rate = target_rate * 1.0  # Mbps
    
        observation_spaces = gym.spaces.Box(low=np.zeros(shape=(self.car_num + 2,), dtype=np.float32),
                                            high=np.ones(shape=(self.car_num + 2,), dtype=np.float32))
        self.observation_space = observation_spaces
    
        action_spaces = gym.spaces.Box(low=-1 * np.ones(shape=(1,), dtype=np.float32),
                                       high=np.ones(shape=(1,), dtype=np.float32))
        self.action_space = action_spaces

    def reset(self):
        self.time = 0  # slot = 1s
        self.buffer.clear()  # reset Buffer
        self.cars_pos_list, self.distance = self.cars_path.load(num_cars=self.car_num)
        self.uav_pos = np.array([220,220,uav_height])

        # Initialize divergence angle
        self.divergence_angle = 0.01 # rad

        state = self.deal_data()
        return state

    def step(self, action):
        self.time += 1
        info = {}
        if self.time < self._max_episode_steps:
            done = False
        else:
            done = True

        # Action: divergence angle theta_tilde - normalized to [-1, 1]
        # Map to [0.001, 0.1] rad range
        self.divergence_angle = 0.001 + (action[0] + 1) / 2 * 0.099
        
        # uav position
        self.uav_pos = uav_pos_fixed.copy()
        
        # Get State and Reward
        state = self.deal_data()
        reward = self.get_reward()

        return state, reward, done, info


    def seed(self, seed=None):
        seed = np.random.seed(seed)
        return [seed]

    def deal_data(self):
        
        # Update car position
        self.cars_pos_list, self.distance = self.cars_path.get_inter_distance(time=self.time,
                                                                                           point=self.uav_pos)

         # Calculate FSO rates
        tx_power_dBm = np.full(len(self.distance), self.p_fso_dBm)
        divergence_angle = np.full(len(self.distance), self.divergence_angle)
        self.rate = get_fso_capacity(tx_power=tx_power_dBm, divergence_angle=divergence_angle, distance=self.distance, car_pos=self.cars_pos_list)
        
        # Calculate optimal divergence 
        self.optimal_divergence = np.array([calculate_optimal_divergence(d, self.wavelength) for d in self.distance])

        # Store data
        self.store()
        
        # State
        dist_norm = self.distance / 500  # Normalize by max distance
        dist_norm = np.clip(dist_norm, 0, 1)

        return dist_norm.astype(np.float32)

    def get_reward(self):
        """Calculate reward"""
        # Calculate rate deviation for each vehicle
        rate_deviation = np.abs(self.rate - self.target_rate) / self.target_rate
        
        # Take max deviation across vehicles
        max_deviation = np.max(rate_deviation)
        
        # Reward function - Eq (22)
        reward = np.exp(-omega * max_deviation) - 1
        
        # Bonus if all vehicles meet target rate
        if np.all(self.rate >= self.target_rate):
            reward += 0.1

        return reward

    def store(self):
        uav = [self.uav_pos]
        car = self.cars_pos_list
        rate = [self.rate]
        # channel = [self.h_fso, self.h_thz]
        self.buffer.update(uav_info=uav, car_info=car, rate_info=rate)

    @property
    def max_episode_steps(self):
        return self._max_episode_steps
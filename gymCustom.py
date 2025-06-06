import gymnasium as gym
from gymnasium import spaces
from gym_utils.reward_typing import RewardFn
import numpy as np
from typing import Union

from gym_utils.Predictor import Predictor


class EngineEnv(gym.Env):
    """
    This is the class that defines the engine environment.

    The observation will be a dictionary with two keys: state and target. The state will be two-dimensional and will
    contain the value of the IMEP and MPRR achieved for a cycle. The target will be one-dimensional and will hold the
    value of the desired IMEP for that cycle.

    observation_space = spaces.Dict(
        "state":
    )

    The action space is of dimension 3 and contains the injection duration, injection pressure, and injection timing.
    """

    metadata = {'render.modes': []}

    def __init__(self,
                 observation_space: spaces.Space = None,
                 action_space: spaces.Space = None,
                 reward: RewardFn = None,
                 ):

        if observation_space is not None:
            self.observation_space = observation_space
        else:
            self.imep_space = np.arange(1.6, 4.1, 0.1)
            self.mprr_space = np.arange(0, 15, 0.5)
            self.observation_space = spaces.Dict(
                {
                    "state": spaces.MultiDiscrete([len(self.imep_space), len(self.mprr_space)]),
                    "target": spaces.Discrete(len(self.imep_space))
                }
            )

        if action_space is not None:
            self.action_space = action_space
        else:
            self.inj_p_space = np.arange(450, 950, 100)
            self.soi_space = np.arange(-5.6, 2.7, 0.1)
            self.inj_d_space = np.arange(0.31, 0.64, 0.01)
            self.action_space = spaces.Tuple((
                # spaces.Discrete(len(self.inj_p_space)),
                spaces.Discrete(len(self.soi_space)),
                spaces.Discrete(len(self.inj_d_space))
            ))
            # self.action_space = spaces.MultiDiscrete([len(self.inj_p_space),
            #                                           len(self.soi_space),
            #                                           len(self.inj_d_space)])

        self.reward = reward

        self._current_mprr = None
        self._current_imep = None
        self._desired_imep = None

        # import torch model
        # define predictor model parameters
        num_layers = 4
        layer_exp = 10
        out_size = 3600  # Need to adjust this based on the format in hdf5 file
        input_size = 3  # number of features
        dropout = 0.1
        self.predictor = Predictor()
        path = '/Users/rodrigohadlich/PycharmProjects/RayProject/AmpereBM/model_weights_mac.pth'
        self.predictor.init_model(input_size, num_layers, layer_exp, out_size, dropout, path)

    def reset(
            self,
            seed: int = None,
            options=None,
    ):
        super().reset(seed=seed)

        # sample random values for observations
        self._current_imep = self.np_random.choice(self.imep_space)
        self._current_mprr = self.np_random.choice(self.mprr_space[self.mprr_space < 7])

        # sample random desired IMEP (must be different from observation)
        while True:
            self._desired_imep = self.np_random.choice(self.imep_space)
            if self._desired_imep != self._current_imep:
                break

        observation = self._get_obs()

        info = None

        return observation, info

    def step(self,
             action_ind: Union[list, np.ndarray] = None,):

        # get actions from indices to values
        inj_p = self.inj_p_space[action_ind[0]]
        soi = self.soi_space[action_ind[1]]
        inj_d = self.inj_d_space[action_ind[2]]
        action_arr = np.array([inj_p, soi, inj_d])

        # print(f"Action is: {action_arr}")

        # send action values to torch model and get new state
        pressure, self._current_imep, self._current_mprr, cad = (
            self.predictor.model_predict(action_arr))

        # get observation in the right format
        observation = self._get_obs()

        # calculate reward
        reward = self.reward(observation)

        # clip observation values to make sure it is within the expected space
        self._current_imep = np.clip(self._current_imep, self.imep_space[0], self.imep_space[-1])
        self._current_mprr = np.clip(self._current_mprr, self.mprr_space[0], self.mprr_space[-1])
        observation = self._get_obs()

        terminated = 0      # will decide in the controller if it is terminated or not
        info = pressure     # return pressure trace cause why not

        return observation, reward, terminated, False, info


    def _get_obs(self):
        return {"state": np.array([self._current_imep, self._current_mprr]), "target": self._desired_imep}


def reward_fn(obs):
    imep, mprr = obs["state"]
    target = obs["target"]

    load_tracking = (imep - target)**2
    safety = max(0, mprr-7)**2

    return -(20*load_tracking + 1000*safety)

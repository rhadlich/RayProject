"""Example of running against a TCP-connected external env performing its own inference.

How to run this script
----------------------
`python [script file name].py --enable-new-api-stack --port 5555

Results to expect
-----------------
You should see something like this on your terminal. Note that the dummy CartPole
client (which runs in a thread for the purpose of this example here) might throw
a disconnection error at the end, b/c RLlib closes the server socket when done training.

+----------------------+------------+--------+------------------+
| Trial name           | status     |   iter |   total time (s) |
|                      |            |        |                  |
|----------------------+------------+--------+------------------+
| PPO_None_3358e_00000 | TERMINATED |     40 |          32.2649 |
+----------------------+------------+--------+------------------+
+------------------------+------------------------+
|  episode_return_mean  |   num_env_steps_sample |
|                       |             d_lifetime |
|-----------------------+------------------------|
|                458.68 |                 160000 |
+-----------------------+------------------------+

"""

from functools import partial
import threading

import gymnasium as gym
import numpy as np

from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from shared_memory_env_runner import SharedMemoryEnvRunner
from ray.rllib.utils.test_utils import (
    add_rllib_example_script_args,
    run_rllib_example_script_experiment,
)
from ray.tune.registry import get_trainable_cls
from multiprocessing import shared_memory

parser = add_rllib_example_script_args(
    default_reward=450.0, default_iters=200, default_timesteps=2000000
)
parser.set_defaults(
    enable_new_api_stack=True,
    num_env_runners=1,
)
parser.add_argument(
    "--weights_shm_name",
    type=str,
    default="weights",
    help="Name of weights shm to use",
)
parser.add_argument(
    "--flag_shm_name",
    type=str,
    default="flag",
    help="Name of flag shm to use",
)

if __name__ == "__main__":
    args = parser.parse_args()

    # Start weights and weights_flag shared memory blocks. Need to instantiate here.
    weight_shape = (256, 128)  # e.g. 256×128 matrix
    weight_dtype = np.float32
    weight_nbytes = np.prod(weight_shape) * np.dtype(weight_dtype).itemsize
    w_shm = shared_memory.SharedMemory(
        create=True,
        name=args.weights_shm_name,
        size=weight_nbytes
    )
    f_shm = shared_memory.SharedMemory(
        create=True,
        name=args.flag_shm_name,
        size=1  # just 1 byte
    )

    # Define name and properties of episode ring buffer to pass down to EnvRunner
    # define the size of each rollout tuple
    bytes_per_float = np.dtype("float32").itemsize      # number of bytes in rollout data type
    dims = {
        "action": 3,
        "reward": 2,
        "state": 4,     # this will be the next state AFTER taking "action"
    }   # the length of the vector of each component of the rollout
    BATCH_SIZE = 32  # number of rollouts per batch (episode)
    NUM_SLOTS = 8  # ring depth
    ELEMENTS_PER_ROLLOUT = sum(dims.values())
    BYTES_PER_ROLLOUT = ELEMENTS_PER_ROLLOUT * bytes_per_float
    # Added state*bytes to PAYLOAD_SIZE because need to include the starting state for each episode. This will be simply
    # the state observation at the end of the last episode/batch.
    PAYLOAD_SIZE = BYTES_PER_ROLLOUT * BATCH_SIZE + dims["state"] * bytes_per_float
    HEADER_SIZE = 8  # write_idx + read_idx (uint32 each)
    HEADER_SLOT_SIZE = 2  # uint16: how many rollouts already in slot
    SLOT_SIZE = HEADER_SLOT_SIZE + PAYLOAD_SIZE
    TOTAL_SIZE = HEADER_SIZE + NUM_SLOTS * SLOT_SIZE

    ep_shm_properties = {
        "BATCH_SIZE": BATCH_SIZE,
        "NUM_SLOTS": NUM_SLOTS,
        "ELEMENTS_PER_ROLLOUT": ELEMENTS_PER_ROLLOUT,
        "BYTES_PER_ROLLOUT": BYTES_PER_ROLLOUT,
        "PAYLOAD_SIZE": PAYLOAD_SIZE,
        "HEADER_SIZE": HEADER_SIZE,
        "HEADER_SLOT_SIZE": HEADER_SLOT_SIZE,
        "SLOT_SIZE": SLOT_SIZE,
        "TOTAL_SIZE": TOTAL_SIZE,
        "STATE_ACTION_DIMS": dims,
        "name": "episodes"
    }

    # Define the RLlib (Master) config.
    base_config = (
        get_trainable_cls(args.algo)
        .get_default_config()
        .environment(
            observation_space=gym.spaces.Box(-1.0, 1.0, (4,), np.float32),
            action_space=gym.spaces.Discrete(2),
            # EnvRunners listen on `port` + their worker index.
            env_config={"weights_shm_name": args.weights_shm_name,
                        "flag_shm_name": args.flag_shm_name,
                        "ep_shm_properties": ep_shm_properties,
                        },
        )
        .env_runners(
            # Point RLlib to the custom EnvRunner to be used here.
            env_runner_cls=SharedMemoryEnvRunner,
        )
        .training(
            num_epochs=10,
            vf_loss_coeff=0.01,
        )
        .rl_module(model_config=DefaultModelConfig(vf_share_layers=True))
    )

    run_rllib_example_script_experiment(base_config, args)

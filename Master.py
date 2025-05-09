"""Example of running against a shared-memory-connected external env performing its own inference.

How to run this script
----------------------
`python master.py --algo "algo_name (like PPO, SAC, etc)" --no-tune

"""

import gymnasium as gym
import numpy as np

from shared_memory_env_runner import SharedMemoryEnvRunner
# from ray.rllib.utils.test_utils import (
#     add_rllib_example_script_args,
#     run_rllib_example_script_experiment,
# )
from ray.tune.registry import get_trainable_cls
from ray.rllib.core.rl_module import RLModuleSpec

from multiprocessing import shared_memory
from define_args import custom_args
from custom_run import run_rllib_shared_memory

parser = custom_args(
    default_reward=450.0, default_iters=200, default_timesteps=2000000
)
parser.set_defaults(
    enable_new_api_stack=True,
    num_env_runners=0,
    num_cpus_per_env_runner=0,
    num_learners=0,
    num_cpus_per_learner=0,
    num_cpus=0,  # for ray_init call inside test_utils
)
parser.add_argument(
    "--policy_shm_name",
    type=str,
    default="policy",
    help="Name of shm for policy weights to use",
)
parser.add_argument(
    "--flag_shm_name",
    type=str,
    default="flag",
    help="Name of flag shm to use",
)

if __name__ == "__main__":
    args = parser.parse_args()

    # Define name and properties of episode ring buffer to pass down to EnvRunner
    # define the size of each rollout tuple
    bytes_per_float = np.dtype("float32").itemsize  # number of bytes in rollout data type
    dims = {
        "action": 3,
        "reward": 1,
        "state": 4,  # this will be the next state AFTER taking "action"
    }  # the length of the vector of each component of the rollout
    BATCH_SIZE = 32  # number of rollouts per batch (episode)
    NUM_SLOTS = 8  # ring depth
    ELEMENTS_PER_ROLLOUT = sum(dims.values())
    BYTES_PER_ROLLOUT = ELEMENTS_PER_ROLLOUT * bytes_per_float
    # Added state*bytes to PAYLOAD_SIZE because need to include the starting state for each episode. This will be simply
    # the state observation at the end of the last episode/batch.
    PAYLOAD_SIZE = ELEMENTS_PER_ROLLOUT * BATCH_SIZE + dims["state"]
    HEADER_SIZE = 2  # write_idx, read_idx
    HEADER_SLOT_SIZE = 1  # one float32 to store how many rollouts already in slot
    SLOT_SIZE = HEADER_SLOT_SIZE + PAYLOAD_SIZE
    TOTAL_SIZE_BYTES = (HEADER_SIZE + NUM_SLOTS * SLOT_SIZE) * bytes_per_float

    ep_shm_properties = {
        "BATCH_SIZE": BATCH_SIZE,
        "NUM_SLOTS": NUM_SLOTS,
        "ELEMENTS_PER_ROLLOUT": ELEMENTS_PER_ROLLOUT,
        "BYTES_PER_ROLLOUT": BYTES_PER_ROLLOUT,
        "PAYLOAD_SIZE": PAYLOAD_SIZE,
        "HEADER_SIZE": HEADER_SIZE,
        "HEADER_SLOT_SIZE": HEADER_SLOT_SIZE,
        "SLOT_SIZE": SLOT_SIZE,
        "TOTAL_SIZE_BYTES": TOTAL_SIZE_BYTES,
        "STATE_ACTION_DIMS": dims,
        "BYTES_PER_FLOAT": bytes_per_float,
        "name": "episodes"
    }

    # Define the RLlib (Master) config.
    base_config = (
        get_trainable_cls(args.algo)
        .get_default_config()
        .environment(
            observation_space=gym.spaces.Box(-1.0, 1.0, (4,), np.float32),
            action_space=gym.spaces.Discrete(2),
            env_config={"policy_shm_name": args.policy_shm_name,
                        "flag_shm_name": args.flag_shm_name,
                        "ep_shm_properties": ep_shm_properties,
                        },
        )
        .env_runners(
            # Point RLlib to the custom EnvRunner to be used here.
            env_runner_cls=SharedMemoryEnvRunner,
            num_env_runners=args.num_env_runners,
            num_cpus_per_worker=args.num_cpus_per_env_runner,
        )
        .training(
            num_epochs=10,
        )
    )

    run_rllib_shared_memory(base_config, args)

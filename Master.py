"""Example of running against a shared-memory-connected external env performing its own inference.

How to run this script
----------------------
`python master.py --algo "algo_name (like PPO, SAC, etc.)" --no-tune

"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from shared_memory_env_runner import SharedMemoryEnvRunner
# from ray.rllib.utils.test_utils import (
#     add_rllib_example_script_args,
#     run_rllib_example_script_experiment,
# )
from ray.tune.registry import get_trainable_cls
from ray.rllib.core.rl_module import RLModuleSpec
from ray.rllib.connectors.env_to_module import FlattenObservations

from multiprocessing import shared_memory
from define_args import custom_args
from custom_run import run_rllib_shared_memory
from impala_debug import IMPALADebug
from ray.rllib.algorithms.impala import IMPALAConfig

import logging
import logging_setup

parser = custom_args(
    default_reward=450.0, default_iters=200, default_timesteps=2000000
)
parser.set_defaults(
    enable_new_api_stack=True,
    num_env_runners=1,  # number of remote EnvRunners
    num_cpus_per_env_runner=1,  # how many cpus per remote EnvRunner
    create_local_env_runner=True,  # only have remote EnvRunners
    create_env_on_local_worker=False,  # don't sample from env if local worker is created
    num_learners=0,  # only have the learner in the main driver
    num_cpus_per_learner=0,  # this will be ignored if num_learners is 0
    num_cpus=3,  # for ray_init call inside test_utils
    algo='IMPALA',
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

    logger = logging.getLogger("MyRLApp.Master")

    # decided to go with one-hot observation representation, manually handling it outside or RLlib.
    # RLlib was having issues with the internal connectors so this was more transparent (and probably faster).

    # define action and observation spaces
    imep_space = np.arange(1.6, 4.1, 0.1)
    mprr_space = np.arange(0, 15, 0.5)
    flat_dim = 2*len(imep_space) + len(mprr_space)
    obs_space = spaces.Box(
            low=0.0, high=1.0, shape=(flat_dim,), dtype=np.float32
        )
    inj_p_space = np.arange(450, 950, 100)
    soi_space = np.arange(-5.6, 2.7, 0.1)
    inj_d_space = np.arange(0.31, 0.64, 0.01)
    action_space = spaces.Tuple((
        # spaces.Discrete(len(inj_p_space)),    # will be kept constant
        spaces.Discrete(len(soi_space)),
        spaces.Discrete(len(inj_d_space))
    ))

    # get action space size in one-hot representation. will need this to create
    # the episode buffer such that it can hold the probability distribution
    action_onehot_size = int(sum([sp.n for sp in action_space]))

    # Define name and properties of episode ring buffer to pass down to EnvRunner
    # define the size of each rollout tuple
    bytes_per_float = np.dtype("float32").itemsize  # number of bytes in rollout data type
    dims = {
        "action": 2,
        "reward": 1,
        "state": 3,                             # this will be the next state AFTER taking "action"
        "action_onehot": action_onehot_size,
        "logp": 1                               # has to be scalar
    }  # the length of the vector of each component of the rollout
    BATCH_SIZE = 32  # number of rollouts per batch (episode)
    NUM_SLOTS = 8  # ring depth
    # increment rollout size to include action probability distribution
    # (size of action one-hot representation) and log probability of
    # selected action (size of action), all in float32 format
    ELEMENTS_PER_ROLLOUT = sum(dims.values())
    BYTES_PER_ROLLOUT = ELEMENTS_PER_ROLLOUT * bytes_per_float
    # Added state to PAYLOAD_SIZE because need to include the starting state for each episode. This will be simply
    # the state observation at the end of the last episode/batch.
    PAYLOAD_SIZE = ELEMENTS_PER_ROLLOUT * BATCH_SIZE + dims["state"]
    HEADER_SIZE = 2  # write_idx, read_idx
    HEADER_SLOT_SIZE = 1  # one float32 to store how many rollouts already in slot
    SLOT_SIZE = HEADER_SLOT_SIZE + PAYLOAD_SIZE
    TOTAL_SIZE = HEADER_SIZE + NUM_SLOTS*SLOT_SIZE
    TOTAL_SIZE_BYTES = int(TOTAL_SIZE * bytes_per_float)
    logger.debug(f"action_onehot_size: {action_onehot_size}")
    logger.debug(f"ELEMENTS_PER_ROLLOUT: {ELEMENTS_PER_ROLLOUT}")
    logger.debug(f"PAYLOAD_SIZE: {PAYLOAD_SIZE}")
    logger.debug(f"SLOT_SIZE: {SLOT_SIZE}")
    logger.debug(f"TOTAL_SIZE: {TOTAL_SIZE}")

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
        "TOTAL_SIZE_BYTES": TOTAL_SIZE_BYTES,
        "STATE_ACTION_DIMS": dims,
        "BYTES_PER_FLOAT": bytes_per_float,
        "name": "episodes",
        "action_onehot_size": action_onehot_size,
    }

    # Define the RLlib (Master) config.
    base_config = (
        get_trainable_cls(args.algo)
        # IMPALAConfig(algo_class=IMPALADebug)
        .get_default_config()
        .api_stack(
            enable_rl_module_and_learner=True,  # turn RLModule on
            enable_env_runner_and_connector_v2=True,  # turn connector-v2 on
        )
        .environment(
            observation_space=obs_space,
            action_space=action_space,
            env_config={"policy_shm_name": args.policy_shm_name,
                        "flag_shm_name": args.flag_shm_name,
                        "ep_shm_properties": ep_shm_properties,
                        "imep_space": imep_space,
                        "mprr_space": mprr_space,
                        },
        )
        .env_runners(
            # Point RLlib to the custom EnvRunner to be used here.
            env_runner_cls=SharedMemoryEnvRunner,
            num_env_runners=args.num_env_runners,
            num_cpus_per_env_runner=args.num_cpus_per_env_runner,
            create_local_env_runner=args.create_local_env_runner,
            create_env_on_local_worker=args.create_env_on_local_worker,
            # env_to_module_connector=_env_to_module_pipeline,    # needed because observation space is a nested structure
        )
        .training(
            num_epochs=10,
            train_batch_size_per_learner=1,
        )
        # .rl_module(rl_module_spec=RLModuleSpec(
        #     observation_space=obs_space,
        #     action_space=action_space,
        #     )
        # )
    )

    run_rllib_shared_memory(base_config, args)

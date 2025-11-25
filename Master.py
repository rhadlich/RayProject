"""Example of running against a shared-memory-connected external env performing its own inference.

How to run this script
----------------------
`python master.py --algo "algo_name (like PPO, SAC, etc.)" --no-tune

"""
import numpy as np
import os

from gymnasium import spaces
from gymCustom import EngineEnvDiscrete, EngineEnvContinuous, reward_fn

from shared_memory_env_runner import SharedMemoryEnvRunner
# from ray.rllib.utils.test_utils import (
#     add_rllib_example_script_args,
#     run_rllib_example_script_experiment,
# )
from ray.tune.registry import get_trainable_cls
from ray.rllib.core.rl_module import RLModuleSpec

from define_args import get_full_parser
from custom_run import run_rllib_shared_memory

from utils import ActionAdapter
from torch_rl_modules.impala_rl_modules import ImpalaMlpModule

import logging

parser = get_full_parser()
parser.set_defaults(
    enable_new_api_stack=True,
    num_env_runners=1,  # number of remote EnvRunners
    num_cpus_per_env_runner=1,  # how many cpus per remote EnvRunner
    create_local_env_runner=True,  # only have remote EnvRunners
    create_env_on_local_worker=False,  # don't sample from env if local worker is created
    num_learners=0,  # only have the learner in the main driver
    num_cpus_per_learner=0,  # this will be ignored if num_learners is 0
    num_cpus=3,  # for ray_init call inside test_utils
    num_gpus_per_learner=0,
    # algo='APPO',
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
    logger.info(f"MASTER, PID={os.getpid()}")

    # make environment to have access to observation and action spaces
    if args.env_type.lower() == 'continuous':
        env = EngineEnvContinuous(reward=reward_fn)
    elif args.env_type.lower() == 'discrete':
        env = EngineEnvDiscrete(reward=reward_fn)
    else:
        raise NotImplementedError(f"Environment type not supported or not provided.")
    obs_space = env.observation_space
    action_space = env.action_space

    if isinstance(obs_space, spaces.Discrete):
        imep_space = env.imep_space
        mprr_space = env.mprr_space

        # patch up the dimensions issue when running a discrete observation space
        flat_dim = len(imep_space)
        obs_space_onehot = spaces.Box(
                low=0.0, high=1.0, shape=(flat_dim,), dtype=np.float32
            )
        obs_is_discrete = True
    elif isinstance(obs_space, spaces.Box):
        obs_space_onehot = None
        imep_space = env.imep_lims
        mprr_space = env.mprr_lims
        obs_is_discrete = False
    else:
        raise NotImplementedError(f"Unsupported observation space {obs_space}")

    adapter = ActionAdapter(action_space)

    # get action space size in one-hot representation (if discrete). will need this to create
    # the episode buffer such that it can hold the probability distribution
    if adapter.mode in ("discrete1", "multidiscrete"):
        # logits will be the same length as action_onehot_size
        action_dist_size = int(sum(adapter.nvec))
    else:
        # non-discrete action space, need mean and standard deviation of the distribution instead of logits
        action_dist_size = 2*action_space.shape[0]

    # Define name and properties of episode ring buffer to pass down to EnvRunner.
    # Define the size of each rollout tuple.
    bytes_per_float = np.dtype("float32").itemsize  # number of bytes in rollout data type
    dims = {
        "action": 2,
        "reward": 1,
        "state": 1,                             # this will be the next state AFTER taking "action"
        "action_dist_size": action_dist_size,
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
    logger.debug(f"action_dist_size: {action_dist_size}")
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
        "action_dist_size": action_dist_size,
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
            observation_space=obs_space_onehot or obs_space,
            action_space=action_space,
            normalize_actions=(True if adapter.mode == "continuous" else False),
            clip_actions=(True if adapter.mode == "continuous" else False),
            clip_rewards=False,
            env_config={"policy_shm_name": args.policy_shm_name,
                        "flag_shm_name": args.flag_shm_name,
                        "ep_shm_properties": ep_shm_properties,
                        "imep_space": imep_space,
                        "mprr_space": mprr_space,
                        "obs_is_discrete": obs_is_discrete,
                        "env_type": args.env_type.lower(),
                        "cpu_core_env_runner": args.cpu_core_env_runner,
                        "cpu_core_minion": args.cpu_core_minion,
                        "enable_zmq": args.enable_zmq,
                        },
        )
        .env_runners(
            # Point RLlib to the custom EnvRunner to be used here.
            env_runner_cls=SharedMemoryEnvRunner,
            num_env_runners=args.num_env_runners,
            num_cpus_per_env_runner=args.num_cpus_per_env_runner,
            create_local_env_runner=args.create_local_env_runner,
            create_env_on_local_worker=args.create_env_on_local_worker,
        )
    )

    import importlib
    try:
        mod = importlib.import_module(f"algo_configs.{args.algo.lower()}_cfg")
        if hasattr(mod, "update_config"):
            base_config = mod.update_config(base_config, args)
    except ModuleNotFoundError:
        pass

    run_rllib_shared_memory(base_config, args)

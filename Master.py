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
from Minion import _minion
from ray.rllib.utils.test_utils import (
    add_rllib_example_script_args,
    run_rllib_example_script_experiment,
)
from ray.tune.registry import get_trainable_cls

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

    # Start the dummy CartPole client in a thread (and do its thing in parallel).
    client_thread = threading.Thread(
        target=partial(
            _minion,
            weights_shm_name=args.weights_shm_name,
            flag_shm_name=args.flag_shm_name,
        ),
    )
    client_thread.start()

    # Define the RLlib (Master) config.
    base_config = (
        get_trainable_cls(args.algo)
        .get_default_config()
        .environment(
            observation_space=gym.spaces.Box(-1.0, 1.0, (4,), np.float32),
            action_space=gym.spaces.Discrete(2),
            # EnvRunners listen on `port` + their worker index.
            env_config={"weights_shm_name": args.weights_shm_name,
                        "flag_shm_name": args.flag_shm_name},
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

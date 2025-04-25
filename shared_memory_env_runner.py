import base64
from collections import defaultdict
import gzip
import json
import pathlib
import socket
import tempfile
import threading
import time
from typing import Collection, DefaultDict, List, Optional, Union

import gymnasium as gym
import numpy as np
import onnxruntime

from ray.rllib.core import (
    Columns,
    COMPONENT_RL_MODULE,
    DEFAULT_AGENT_ID,
    DEFAULT_MODULE_ID,
)
from ray.rllib.env import INPUT_ENV_SPACES
from ray.rllib.env.env_runner import EnvRunner
from ray.rllib.env.single_agent_env_runner import SingleAgentEnvRunner
from ray.rllib.env.single_agent_episode import SingleAgentEpisode
from ray.rllib.env.utils.external_env_protocol import RLlink as rllink
from ray.rllib.utils.annotations import ExperimentalAPI, override
from ray.rllib.utils.checkpoints import Checkpointable
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.metrics import (
    EPISODE_DURATION_SEC_MEAN,
    EPISODE_LEN_MAX,
    EPISODE_LEN_MEAN,
    EPISODE_LEN_MIN,
    EPISODE_RETURN_MAX,
    EPISODE_RETURN_MEAN,
    EPISODE_RETURN_MIN,
    WEIGHTS_SEQ_NO,
)
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger
from ray.rllib.utils.numpy import softmax
from ray.rllib.utils.typing import EpisodeID, StateDict

from multiprocessing import shared_memory


class SharedMemoryEnvRunner(EnvRunner):
    """An EnvRunner communicating with an external env through shared memory.

        This implementation assumes:
        - Only one external client ever connects to this env runner.
        - The external client performs inference locally through an ONNX model. Thus,
        samples are sent in bulk once a certain number of timesteps has been executed on the
        client's side (no individual action requests).
        - A copy of the RLModule is kept at all times on the env runner, but never used
        for inference, only as a data (weights) container.
        TODO (sven): The above might be inefficient as we have to store basically two
         models, one in this EnvRunner, one in the env (as ONNX).
        - There is no environment and no connectors on this env runner. The external env
        is responsible for generating all the data to create episodes.
        """

    def __init__(self, *, config, **kwargs):
        """
        Initializes a SharedMemoryEnvRunner instance.

        Args:
            config: The AlgorithmConfig to use for setup.

        Keyword Args:
            weights_shm_name: name of the shared memory block that will contain model weights
            flag_shm_name: name of the shared memory block that will contain update flag
        """
        super().__init__(config=config)

        self.worker_index: int = kwargs.get("worker_index", 0)

        self._weights_seq_no = 0

        # Build the module from its spec.
        module_spec = self.config.get_rl_module_spec(
            spaces=self.get_spaces(), inference_only=True
        )
        self.module = module_spec.build()

        # communication settings go here
        self.weights_shm_name = self.config.env_config.get("weights_shm_name", 'weight')
        self.flag_shm_name = self.config.env_config.get("flag_shm_name", 'flag')
        self.weights_shm = None
        self.flag_shm = None

        self.metrics = MetricsLogger()

        self._episode_chunks_to_return: Optional[List[SingleAgentEpisode]] = None
        self._done_episodes_for_metrics: List[SingleAgentEpisode] = []
        self._ongoing_episodes_for_metrics: DefaultDict[
            EpisodeID, List[SingleAgentEpisode]
        ] = defaultdict(list)

        self._sample_lock = threading.Lock()
        self._on_policy_lock = threading.Lock()
        self._blocked_on_state = False

        # # Start a background thread for client communication.
        # self.thread = threading.Thread(
        #     target=self._client_message_listener, daemon=True
        # )
        # self.thread.start()

    def assert_healthy(self):
        """Checks that the server socket is open and listening."""
        assert (
                self.weights_shm is not None
        ), "Server socket is None (not connected, not listening)."

    def sample(self, **kwargs):
        """Waits for the client to send episodes."""
        while True:
            with self._sample_lock:
                if self._episode_chunks_to_return is not None:
                    num_env_steps = 0
                    num_episodes_completed = 0
                    for eps in self._episode_chunks_to_return:
                        if eps.is_done:
                            self._done_episodes_for_metrics.append(eps)
                            num_episodes_completed += 1
                        else:
                            self._ongoing_episodes_for_metrics[eps.id_].append(eps)
                        num_env_steps += len(eps)

                    ret = self._episode_chunks_to_return
                    self._episode_chunks_to_return = None

                    SingleAgentEnvRunner._increase_sampled_metrics(
                        self, num_env_steps, num_episodes_completed
                    )

                    return ret
            time.sleep(0.01)        #! is this necessary??

    def get_metrics(self):
        # Compute per-episode metrics (only on already completed episodes).
        for eps in self._done_episodes_for_metrics:
            assert eps.is_done
            episode_length = len(eps)
            episode_return = eps.get_return()
            episode_duration_s = eps.get_duration_s()
            # Don't forget about the already returned chunks of this episode.
            if eps.id_ in self._ongoing_episodes_for_metrics:
                for eps2 in self._ongoing_episodes_for_metrics[eps.id_]:
                    episode_length += len(eps2)
                    episode_return += eps2.get_return()
                    episode_duration_s += eps2.get_duration_s()
                del self._ongoing_episodes_for_metrics[eps.id_]

            self._log_episode_metrics(
                episode_length, episode_return, episode_duration_s
            )

        # Now that we have logged everything, clear cache of done episodes.
        self._done_episodes_for_metrics.clear()

        # Return reduced metrics.
        return self.metrics.reduce()

    def get_spaces(self):
        return {
            INPUT_ENV_SPACES: (self.config.observation_space, self.config.action_space),
            DEFAULT_MODULE_ID: (
                self.config.observation_space,
                self.config.action_space,
            ),
        }

    def stop(self):
        """Closes and unlinks the weights and flag shared memory blocks."""
        self._unlink_mem_blocks_if_necessary()

    def get_ctor_args_and_kwargs(self):
        return (
            (),  # *args
            {"config": self.config},  # **kwargs
        )

    def get_checkpointable_components(self):
        return [
            (COMPONENT_RL_MODULE, self.module),
        ]

    def get_state(
            self,
            components: Optional[Union[str, Collection[str]]] = None,
            *,
            not_components: Optional[Union[str, Collection[str]]] = None,
            **kwargs,
    ) -> StateDict:
        return {}

    def set_state(self, state: StateDict) -> None:
        # Update the RLModule state.
        if COMPONENT_RL_MODULE in state:
            # A missing value for WEIGHTS_SEQ_NO or a value of 0 means: Force the
            # update.
            weights_seq_no = state.get(WEIGHTS_SEQ_NO, 0)

            # Only update the weigths, if this is the first synchronization or
            # if the weights of this `EnvRunner` lacks behind the actual ones.
            if weights_seq_no == 0 or self._weights_seq_no < weights_seq_no:
                rl_module_state = state[COMPONENT_RL_MODULE]
                if (
                        isinstance(rl_module_state, dict)
                        and DEFAULT_MODULE_ID in rl_module_state
                ):
                    rl_module_state = rl_module_state[DEFAULT_MODULE_ID]
                self.module.set_state(rl_module_state)

            # Update our weights_seq_no, if the new one is > 0.
            if weights_seq_no > 0:
                self._weights_seq_no = weights_seq_no

        if self._blocked_on_state is True:
            self._send_set_state_message()
            self._blocked_on_state = False



    def _unlink_mem_blocks_if_necessary(self):
        if self.weights_shm:
            self.weights_shm.close()
            self.weights_shm.unlink()
        if self.flag_shm:
            self.flag_shm.close()
            self.flag_shm.unlink()

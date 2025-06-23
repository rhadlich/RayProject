from collections import defaultdict
from typing import DefaultDict, List, Optional, Union, Collection

import gymnasium as gym
import numpy as np
import torch
import os
import gzip
import time

from ray.rllib.core import (
    COMPONENT_RL_MODULE,
    DEFAULT_AGENT_ID,
    DEFAULT_MODULE_ID,
    DEFAULT_POLICY_ID,
    COMPONENT_ENV_TO_MODULE_CONNECTOR,
    COMPONENT_MODULE_TO_ENV_CONNECTOR
)
from ray.rllib.env import INPUT_ENV_SPACES, INPUT_ENV_SINGLE_SPACES
from ray.rllib.env.env_runner import EnvRunner
from ray.rllib.utils.framework import get_device
from ray.rllib.utils.checkpoints import Checkpointable
from ray.rllib.env.single_agent_env_runner import SingleAgentEnvRunner
from ray.rllib.env.single_agent_episode import SingleAgentEpisode
from ray.rllib.utils.metrics import (
    EPISODE_DURATION_SEC_MEAN,
    EPISODE_LEN_MAX,
    EPISODE_LEN_MEAN,
    EPISODE_LEN_MIN,
    NUM_ENV_STEPS_SAMPLED_LIFETIME,
    EPISODE_RETURN_MAX,
    EPISODE_RETURN_MEAN,
    EPISODE_RETURN_MIN,
    WEIGHTS_SEQ_NO,
)
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger
from ray.rllib.utils.typing import EpisodeID, StateDict, PolicyID

from ray.rllib.policy.policy_map import PolicyMap
from ray.rllib.policy.policy import Policy

from multiprocessing import shared_memory
import minion
from custom_run import _get_current_onnx_model
import struct
import ray
from minion import (
    get_indices,
    set_indices,
    flatten_obs_onehot
)

import logging
import logging_setup
from pprint import pformat


@ray.remote(num_cpus=1)
def _run_minion(policy_shm_name: str,
                flag_shm_name: str,
                ep_shm_name: str,
                episode_shm_properties: dict,
                logger,
                ) -> None:
    """
    Thin wrapper that just calls minion to execute in a separate CPU core.
    """
    logger.debug("IN _RUN_MINION NOW.")
    logger.debug(f"_run_minion called by PID: {os.getpid()}")
    minion.main(policy_shm_name, flag_shm_name, ep_shm_name, episode_shm_properties)


def _decode_obs(obs):
    """
    Function that takes in flattened observation (np.array) and converts it to
    the format of the environment's observation space.
    """
    # return {"state": obs[:-1], "target": obs[-1]}
    return obs


class SharedMemoryEnvRunner(EnvRunner, Checkpointable):
    """An EnvRunner communicating with an external env through shared memory.

        This implementation assumes:
        - Only one external client ever connects to this env runner.
        - The external client performs inference locally through an ONNX model. Thus,
        samples are sent in bulk once a certain number of timesteps has been executed on the
        client's side (no individual action requests).
        - A copy of the RLModule is kept at all times on the env runner, but never used
        for inference, only as a data (weights) container.
        - There is no environment and no connectors on this env runner. The external env
        is responsible for generating all the data to create episodes.
        - Can only be used for off-policy algorithms. No blocking condition has been created
        to stop the client from collecting new episodes before receiving updated weights.
        """

    def __init__(self, *, config, **kwargs):
        """
        Initializes a SharedMemoryEnvRunner instance.

        Args:
            config: The AlgorithmConfig to use for setup (set by RLlib).
            worker_index: The index of the worker process to use (set by RLlib).
            num_workers: The number of worker processes to use (set by RLlib).

        Keyword Args:
            policy_shm_name: name of the shared memory block that will contain model weights
            flag_shm_name: name of the shared memory block that will contain update flag
        """
        super().__init__(config=config)

        self.logger = logging.getLogger("MyRLApp.EnvRunner")
        self.logger.info(f"SharedMemoryEnvRunner, PID={os.getpid()}")

        self.worker_index: int = kwargs.get("worker_index")
        self.spaces = kwargs.get("spaces", {})

        # Set device.
        self._device = get_device(
            self.config,
            0 if not self.worker_index else self.config.num_gpus_per_env_runner,
        )

        # dummy method to populate self.spaces to stay consistent with SingleAgentEnvRunner
        self.make_env()

        self.logger.debug(f"EnvRunner: spaces -> {self.spaces}")

        # from SingleAgentEnvRunner
        # Create the env-to-module connector pipeline.
        self._env_to_module = self.config.build_env_to_module_connector(
            env=self.env, spaces=self.spaces, device=self._device
        )
        # Create the module-to-env connector pipeline.
        self._module_to_env = self.config.build_module_to_env_connector(
            env=self.env, spaces=self.spaces
        )

        self.logger.debug("EnvRunner: Started __init__()")

        self.worker_index: int = kwargs.get("worker_index", 0)

        self._weights_seq_no = 0
        self.weight_update_no = 0

        self.logger.debug("EnvRunner: obs and act spaces:")
        self.logger.debug(f"{(self.config.observation_space, self.config.action_space)}")

        # Build the module from its spec.
        module_spec = self.config.get_rl_module_spec(
            spaces=self.get_spaces(), inference_only=True
        )
        self.module = module_spec.build()

        self.metrics = MetricsLogger()

        self._episode_chunks_to_return: Optional[List[SingleAgentEpisode]] = None
        self._done_episodes_for_metrics: List[SingleAgentEpisode] = []
        self._ongoing_episodes_for_metrics: DefaultDict[
            EpisodeID, List[SingleAgentEpisode]
        ] = defaultdict(list)

        # Policy weights communication stuff goes here
        self.policy_shm_name = self.config.env_config.get("policy_shm_name", 'policy')
        self.flag_shm_name = self.config.env_config.get("flag_shm_name", 'flag')
        # Can't get shm references in init because the run method only creates the shm blocks once init is done
        self.policy_shm = False
        self.flag_shm = False

        # get dimensions of observation space
        self.imep_space = self.config.env_config["imep_space"]
        self.mprr_space = self.config.env_config["mprr_space"]

        # Block some things from the EnvRunner execution in driver process.
        self._local_worker_bool = (self.worker_index == 0 and self.config.num_env_runners > 0)
        if not self._local_worker_bool:
            # Start shared memory block for episodes
            self.BATCH_SIZE = self.config.env_config["ep_shm_properties"]["BATCH_SIZE"]
            self.NUM_SLOTS = self.config.env_config["ep_shm_properties"]["NUM_SLOTS"]
            self.ELEMENTS_PER_ROLLOUT = self.config.env_config["ep_shm_properties"]["ELEMENTS_PER_ROLLOUT"]
            self.BYTES_PER_ROLLOUT = self.config.env_config["ep_shm_properties"]["BYTES_PER_ROLLOUT"]
            self.PAYLOAD_SIZE = self.config.env_config["ep_shm_properties"]["PAYLOAD_SIZE"]
            self.HEADER_SIZE = self.config.env_config["ep_shm_properties"]["HEADER_SIZE"]
            self.HEADER_SLOT_SIZE = self.config.env_config["ep_shm_properties"]["HEADER_SLOT_SIZE"]
            self.SLOT_SIZE = self.config.env_config["ep_shm_properties"]["SLOT_SIZE"]
            self.TOTAL_SIZE = self.config.env_config["ep_shm_properties"]["TOTAL_SIZE"]
            self.TOTAL_SIZE_BYTES = self.config.env_config["ep_shm_properties"]["TOTAL_SIZE_BYTES"]
            self.STATE_ACTION_DIMS = self.config.env_config["ep_shm_properties"]["STATE_ACTION_DIMS"]
            self.BYTES_PER_FLOAT = self.config.env_config["ep_shm_properties"]["BYTES_PER_FLOAT"]
            self.action_onehot_size = self.config.env_config["ep_shm_properties"]["action_onehot_size"]
            self.episode_shm = shared_memory.SharedMemory(
                create=True,
                name=self.config.env_config["ep_shm_properties"].get("name", 'episodes'),
                size=self.TOTAL_SIZE_BYTES
            )
            self.ep_buf = self.episode_shm.buf
            self.ep_arr = np.ndarray(shape=(self.TOTAL_SIZE,),
                                     dtype=np.float32,
                                     buffer=self.ep_buf
                                     )
            self.logger.debug(f"ep_arr size -> {self.ep_arr.shape}")
            self.logger.debug(f"EnvRunner: e_arr is writable? {self.ep_arr.flags.writeable}.")
            self.episode_shm_name = self.episode_shm.name

            self.logger.debug("EnvRunner: Created episode_shm.")

            # Initialize the write and read indices in the episode buffer to 0
            self.ep_arr[:2] = 0

            # Spawn minion subprocess. Decided to go with ray task because it's convenient. This approach can split the
            # workload in the Raspberry Pi to be: 1 core Algorithm/Learner, 1 core EnvRunner, 1 core Minion
            # (environment), and the other core for the OS and any other support tasks.
            self._minion_ref = _run_minion.remote(
                self.policy_shm_name,
                self.flag_shm_name,
                self.episode_shm_name,
                self.config.env_config["ep_shm_properties"],
                self.logger,
            )

            # code will not run without this, something to do with timing of when processes get spawned
            ready, _ = ray.wait([self._minion_ref], timeout=1.0)
            self.logger.debug(f"EnvRunner: minion scheduled? {bool(ready)}")
            self.logger.debug("EnvRunner: Spawned minion. Done with __init__")

            # Set first_assert_healthy flag to true. This was needed due to a race condition in creating/accessing the
            # policy shared memory block
            self.first_assert_healthy = True

    def assert_healthy(self):
        """Checks that shared memory blocks and minion process have been created."""
        self.logger.debug("EnvRunner: Called assert_healthy()")

        minion_running, _ = ray.wait([self._minion_ref], timeout=0)
        if not minion_running:
            minion_running = None

        if not self.first_assert_healthy:
            # in the second iteration, get shm references
            if self.policy_shm == False:
                self.logger.debug("EnvRunner: Calling _get_shm_references() in assert_healthy()")
                self._get_shm_references()

            assert (
                    self.policy_shm is not None and
                    self.flag_shm is not None and
                    self.episode_shm is not None and
                    minion_running is not None
            ), "Shared memory blocks or minion process not alive."
        else:
            # set flag to false so it checks
            self.first_assert_healthy = False

    def sample(self, **kwargs):
        """
        Get new training batches.
        Returns an empty list if no new batches are available.
        """
        # self.logger.debug("EnvRunner: Called sample()")

        episodes = None

        # block until a new episode is available
        while episodes is None:
            try:
                episodes = self._read_batch()
            except Exception as e:
                self.logger.error(f"EnvRunner: Exception reading episodes: {e}")
            if episodes is None:
                time.sleep(0.0001)

        # if new batches were found
        num_env_steps = 0
        num_episodes_completed = 0
        for eps in episodes:
            assert eps.is_done
            self._done_episodes_for_metrics.append(eps)
            num_episodes_completed += 1
            num_env_steps += len(eps)

            # # checks for dimensions
            # try:
            #     obs = np.array(eps.get_observations())
            #     actions = np.array(eps.get_actions())
            #     rewards = np.array(eps.get_rewards())
            #     self.logger.debug(f"EnvRunner(sample): observations -> {obs.shape}")
            #     self.logger.debug(f"EnvRunner(sample): actions -> {actions.shape}")
            #     self.logger.debug(f"EnvRunner(sample): rewards -> {rewards.shape}")
            # except Exception as e:
            #     self.logger.debug(f"EnvRunner(sample): Could not get episode dimensions due to error {e}")

        # not sure why/if this is necessary but TCP example does it so couldn't hurt
        SingleAgentEnvRunner._increase_sampled_metrics(
            self, num_env_steps, num_episodes_completed
        )

        # self.logger.debug(f"EnvRunner(sample): Logged {len(episodes)} episodes.")

        return episodes

    def get_metrics(self):
        # Compute per-episode metrics (only on already completed episodes).
        # self.logger.debug("EnvRunner: Called get_metrics()")

        # self.logger.debug(f"EnvRunner(get_metrics): weights_seq_no -> {self._weights_seq_no}.")

        if not self._done_episodes_for_metrics:
            # self.logger.debug("EnvRunner(get_metrics): No new episodes to compute metrics.")
            return {}

        for eps in self._done_episodes_for_metrics:
            assert eps.is_done
            episode_length = len(eps)
            episode_return = eps.get_return()
            episode_duration_s = eps.get_duration_s()

            # self.logger.debug(f"EnvRunner(get_metrics): episode length, return, and duration: "
            #                   f"{episode_length}, {episode_return}, {episode_duration_s}")

            self._log_episode_metrics(
                episode_length, episode_return, episode_duration_s
            )

        # Now that we have logged everything, clear cache of done episodes.
        self._done_episodes_for_metrics.clear()

        # self.logger.debug("EnvRunner(get_metrics): Logged new episode metrics.")

        # Return reduced metrics.
        return self.metrics.reduce()

    def stop(self):
        """Closes and unlinks the weights, flag, and episode shared memory blocks."""
        self.logger.debug("EnvRunner: Called stop()")
        self._unlink_mem_blocks_if_necessary()

    def get_spaces(self):
        self.logger.debug(f"EnvRunner: Called get_spaces(), PID={os.getpid()}")
        return {
            INPUT_ENV_SPACES: (self.config.observation_space, self.config.action_space),
            DEFAULT_MODULE_ID: (
                self.config.observation_space,
                self.config.action_space,
            ),
        }

    def get_state(
        self,
        components: Optional[Union[str, Collection[str]]] = None,
        *,
        not_components: Optional[Union[str, Collection[str]]] = None,
        **kwargs,
    ) -> StateDict:

        # self.logger.debug("EnvRunner: Called get_state()")

        state = {
            NUM_ENV_STEPS_SAMPLED_LIFETIME: (
                self.metrics.peek(NUM_ENV_STEPS_SAMPLED_LIFETIME, default=0)
            ),
        }

        if self._check_component(COMPONENT_RL_MODULE, components, not_components):
            state[COMPONENT_RL_MODULE] = self.module.get_state(
                components=self._get_subcomponents(COMPONENT_RL_MODULE, components),
                not_components=self._get_subcomponents(
                    COMPONENT_RL_MODULE, not_components
                ),
                **kwargs,
            )
            state[WEIGHTS_SEQ_NO] = self._weights_seq_no
        if self._check_component(
            COMPONENT_ENV_TO_MODULE_CONNECTOR, components, not_components
        ):
            state[COMPONENT_ENV_TO_MODULE_CONNECTOR] = self._env_to_module.get_state()
        if self._check_component(
            COMPONENT_MODULE_TO_ENV_CONNECTOR, components, not_components
        ):
            state[COMPONENT_MODULE_TO_ENV_CONNECTOR] = self._module_to_env.get_state()

        # self.logger.debug(f"EnvRunner (get_state): NUM_ENV_STEPS_SAMPLED_LIFETIME={state[NUM_ENV_STEPS_SAMPLED_LIFETIME]}")

        return state

    def set_state(self, state: StateDict) -> None:
        self.logger.debug("EnvRunner: Called set_state()")
        # Update the RLModule state.
        if COMPONENT_RL_MODULE in state:
            # A missing value for WEIGHTS_SEQ_NO or a value of 0 means: Force the
            # update.
            weights_seq_no = state.get(WEIGHTS_SEQ_NO, 0)

            self.logger.debug(f"EnvRunner: weights_seq_no={weights_seq_no}")

            # Only update the weights, if this is the first synchronization or
            # if the weights of this `EnvRunner` lacks behind the actual ones.
            if weights_seq_no == 0 or self._weights_seq_no < weights_seq_no:
                rl_module_state = state[COMPONENT_RL_MODULE]
                # this was needed because ray was passing an object reference after
                # the first iteration to decrease latency. need to dereference
                if isinstance(rl_module_state, ray.ObjectRef):
                    rl_module_state = ray.get(rl_module_state)

                if (
                    isinstance(rl_module_state, dict)
                    and DEFAULT_MODULE_ID in rl_module_state
                ):
                    rl_module_state = rl_module_state[DEFAULT_MODULE_ID]

                self.module.set_state(rl_module_state)

                # send new weights to minion
                if self.worker_index > 0:
                    # make sure shared memory blocks have been linked
                    if self.policy_shm == False:
                        self._get_shm_references()

                    # get new model weights
                    ort_raw = _get_current_onnx_model(self.module, logger=self.logger)

                    # update the policy weights in the policy shared memory buffer
                    self._update_policy_shm(ort_raw)

                    # set weights-available flag to 1 (true) so that minion can update
                    self.f_buf[1] = 1

                    self.weight_update_no += 1
                    self.logger.debug(f"EnvRunner: number of weight updates -> {self.weight_update_no}")

            # Update our weights_seq_no, if the new one is > 0.
            if weights_seq_no > 0:
                self._weights_seq_no = weights_seq_no

            # Update our lifetime counters.
            if NUM_ENV_STEPS_SAMPLED_LIFETIME in state:
                self.metrics.set_value(
                    key=NUM_ENV_STEPS_SAMPLED_LIFETIME,
                    value=state[NUM_ENV_STEPS_SAMPLED_LIFETIME],
                    reduce="sum",
                    with_throughput=True,
                )

    def make_env(self):
        # Do NOT instantiate a Gym env – sampling happens in the minion.
        self.env = None
        self.num_envs = 0
        self.spaces = {
            INPUT_ENV_SINGLE_SPACES: (self.config.observation_space,
                                      self.config.action_space)
        }

    def get_ctor_args_and_kwargs(self):
        return (
            (),  # *args
            {"config": self.config},  # **kwargs
        )

    # Helper functions
#   #------------------------------------------------------------------------#
    def _get_shm_references(self):
        while True:
            try:
                # get reference to policy shared memory weights
                self.policy_shm = shared_memory.SharedMemory(name=self.policy_shm_name, create=False)
                self.flag_shm = shared_memory.SharedMemory(name=self.flag_shm_name, create=False)
                self.p_buf = self.policy_shm.buf
                self.f_buf = self.flag_shm.buf
                self.logger.debug("EnvRunner: Created shm references.")
                break
            except FileNotFoundError:
                # could reach here before Master has created the shm. if so, wait and try again
                time.sleep(0.05)

    def _update_policy_shm(self, ort_raw: bytes):
        self.logger.debug("EnvRunner: Called update_policy_shm()")

        # get expected ort_compressed length from header
        _len_ort_expected = struct.unpack_from("<I", self.p_buf, 0)
        header_offset = 4   # number of bytes in the int

        # sanity check to make sure the size of the weights vector is as expected
        assert _len_ort_expected[0] == len(ort_raw), (
            f"Expected model size {_len_ort_expected[0]} bytes but got {len(ort_raw)} bytes."
        )

        # if lock is set to true wait until it isn't and then update the weights
        while self.f_buf[0] == 1:
            time.sleep(0.0001)

        # save new model weights to buffer
        self.f_buf[0] = 1  # set lock flag to locked
        self.p_buf[header_offset:header_offset+len(ort_raw)] = ort_raw  # insert compressed weights
        self.f_buf[0] = 0  # set lock flag to unlocked
        self.f_buf[1] = 1  # set weights-available flag to 1 (true)

    def _unlink_mem_blocks_if_necessary(self):
        """Close all shared memory blocks and terminate minion subprocess."""
        if self.policy_shm:
            self.policy_shm.close()
            self.policy_shm.unlink()
        if self.flag_shm:
            self.flag_shm.close()
            self.flag_shm.unlink()
        if self.episode_shm:
            del self.ep_arr
            self.episode_shm.close()
            self.episode_shm.unlink()
        if ray.wait([self._minion_ref], timeout=0)[0]:
            ray.cancel(self._minion_ref)

    def _log_episode_metrics(self, length, ret, sec):
        # Log general episode metrics.
        # To mimic the old API stack behavior, we'll use `window` here for
        # these particular stats (instead of the default EMA).
        win = self.config.metrics_num_episodes_for_smoothing        # default is 100 episodes for smoothing
        self.metrics.log_value(EPISODE_LEN_MEAN, length, window=win)
        self.metrics.log_value(EPISODE_RETURN_MEAN, ret, window=win)
        self.metrics.log_value(EPISODE_DURATION_SEC_MEAN, sec, window=win)
        # Per-agent returns.
        self.metrics.log_value(
            ("agent_episode_returns_mean", DEFAULT_AGENT_ID), ret, window=win
        )
        # Per-RLModule returns.
        self.metrics.log_value(
            ("module_episode_returns_mean", DEFAULT_MODULE_ID), ret, window=win
        )

        # For some metrics, log min/max as well.
        self.metrics.log_value(EPISODE_LEN_MIN, length, reduce="min", window=win)
        self.metrics.log_value(EPISODE_RETURN_MIN, ret, reduce="min", window=win)
        self.metrics.log_value(EPISODE_LEN_MAX, length, reduce="max", window=win)
        self.metrics.log_value(EPISODE_RETURN_MAX, ret, reduce="max", window=win)

    # Helper functions for episode ring buffer manipulation
    def _read_batch(self):
        """
        Pop the oldest COMPLETE 32-rollout batch from the ring.
        Returns list of SingleAgentEpisode objects or None if no complete slot is ready.

        Structure of the ring is as follows:
        ┌────────────────────────────────────────────────────────────────┐
        │ Offset 0                                                       │
        │ ┌──────────────────────┐                                       │
        │ │ write_idx (uint32)   │   ← head pointer (next slot to write) │
        │ └──────────────────────┘                                       │
        │ ┌──────────────────────┐                                       │
        │ │ read_idx  (uint32)   │   ← tail pointer (next slot to read)  │
        │ └──────────────────────┘                                       │
        │ Offset 8                                                       │
        │ ┌─────────────────────────────────────────────────────────────┐│
        │ │ Slot 0 (SLOT_SIZE bytes):                                   ││
        │ │   ┌────────────────┐  ┌──────────────────────────────────┐  ││
        │ │   │ length (uint16)│  │ payload bytes (≤ PAYLOAD_SIZE)   │  ││
        │ │   └────────────────┘  └──────────────────────────────────┘  ││
        │ └─────────────────────────────────────────────────────────────┘│
        │ ┌─────────────────────────────────────────────────────────────┐│
        │ │ Slot 1 (SLOT_SIZE bytes): …                                 ││
        │ └─────────────────────────────────────────────────────────────┘│
        │                              …                                 │
        │ ┌─────────────────────────────────────────────────────────────┐│
        │ │ Slot N-1                                                    ││
        │ └─────────────────────────────────────────────────────────────┘│
        └────────────────────────────────────────────────────────────────┘

        The structure for each slot is:
        Slot k (size = HEADER_SLOT_SIZE + PAYLOAD_SIZE)
        ┌──────────────────────────────────────────────────────┐
        │ HEADER_SLOT_SIZE bytes:                              │
        │   • filled_count  (uint16)      ← how many rollouts  │
        ├──────────────────────────────────────────────────────┤
        │ PAYLOAD_SIZE bytes:                                  │
        │   ┌ starting_state (state_dim floats)                │
        │   ├ rollout[0]:                                      │
        │   │   ┌ action    (state_dim floats)                 │
        │   │   ├ reward    (action_dim floats)                │
        │   │   └ state     (state_dim floats)                 │
        │   ├ rollout[1]:  (same layout)                       │
        ├ ...                                                  │
        │   └ rollout[batch_size − 1]:                         │
        │       └ same as rollout[0]                           │
        └──────────────────────────────────────────────────────┘
        """
        # self.logger.debug("EnvRunner: In _read_batch().")

        # wait until the buffer is unlocked to read indices, then read and lock (locking happens in get_indices)
        while True:
            if self.f_buf[3] == 0:
                write_idx, read_idx = get_indices(self.ep_arr, self.f_buf)
                # self.logger.debug(f"EnvRunner(_read_batch): Started reading buffer. "
                #                   f"Identified write_idx, read_idx: {write_idx}, {read_idx}.")
                break
            else:
                time.sleep(0.0001)

        # if any of these are true it means the last batch read is the most updated full batch
        if write_idx == read_idx:
            # self.logger.debug("EnvRunner(_read_batch): Returning None.")
            set_indices(self.ep_arr, read_idx, 'r', self.f_buf)   # to unlock episode buffer
            # self.logger.debug(f"EnvRunner(_read_batch): Done reading buffer, no episodes available. "
            #                   f"Final read index: {read_idx}.")
            return None  # ring empty

        # In case the write_idx has wrapped around and there are more full batches available
        elif write_idx < read_idx:
            # write_idx + 1 because indexing starts at 0.
            num_batches = ((self.NUM_SLOTS-1) - read_idx) + write_idx + 1

        # regular case where there are full batches available
        else:
            num_batches = write_idx - read_idx

        # self.logger.debug(f"EnvRunner(_read_batch): num_batches: {num_batches}. Going into batches loop.")

        batches = []
        for i in range(num_batches):
            # assert that the batch/slot is full
            slot_off = self.HEADER_SIZE + read_idx * self.SLOT_SIZE
            filled = int(self.ep_arr[slot_off])
            assert filled == self.BATCH_SIZE

            # extract data from ring
            data_start = slot_off + self.HEADER_SLOT_SIZE
            payload = np.copy(self.ep_arr[data_start: data_start + self.PAYLOAD_SIZE])  # extract data from array

            # extract initial state from front of payload and remove it so that shape is consistent for next step
            initial_state = payload[:self.STATE_ACTION_DIMS["state"]]
            payload = np.delete(payload, np.arange(self.STATE_ACTION_DIMS["state"]))

            # reshape data and add truncated and terminated fields
            payload = payload.reshape(self.BATCH_SIZE, self.ELEMENTS_PER_ROLLOUT)
            terminateds = np.zeros(self.BATCH_SIZE, dtype=np.bool_)
            truncateds = np.zeros(self.BATCH_SIZE, dtype=np.bool_)
            truncateds[-1] = True

            # instantiate batch, format with SingleAgentEpisode
            batch = SingleAgentEpisode(
                observation_space=self.config.observation_space,
                action_space=self.config.action_space,
            )

            # add initial state
            batch.add_env_reset(flatten_obs_onehot(_decode_obs(initial_state), self.imep_space, self.mprr_space))

            # get start and end indices of each component of the payload
            action_start = 0
            action_end = self.STATE_ACTION_DIMS["action"]
            reward_start = action_end
            reward_end = reward_start + self.STATE_ACTION_DIMS["reward"]
            obs_start = reward_end
            obs_end = obs_start + self.STATE_ACTION_DIMS["state"]
            logp_start = obs_end
            logp_end = logp_start + self.STATE_ACTION_DIMS["logp"]
            dist_start = logp_end
            dist_end = dist_start + self.STATE_ACTION_DIMS["action_onehot"]

            # add one rollout at a time (i.e. one row in payload)
            for j, rollout in enumerate(payload):
                batch.add_env_step(
                    action=rollout[action_start:action_end],
                    reward=np.squeeze(rollout[reward_start:reward_end]),
                    # observation !AFTER! taking "action"
                    observation=flatten_obs_onehot(_decode_obs(rollout[obs_start:obs_end]),
                                                    self.imep_space, self.mprr_space),
                    terminated=bool(terminateds[j]),
                    truncated=bool(truncateds[j]),
                    extra_model_outputs={
                        "action_logp": np.squeeze(rollout[logp_start:logp_end]),
                        "action_dist_inputs": rollout[dist_start:dist_end],
                    }
                )

            # append to list of batches to pass to learner
            batches.append(batch.to_numpy())

            # Advance read_idx
            read_idx = (read_idx + 1) % self.NUM_SLOTS

        # self.logger.debug("EnvRunner(_read_batch): Done logging batches.")

        # commit new indices and unlock episode buffer (unlocking happens inside set_indices)
        set_indices(self.ep_arr, read_idx, 'r', self.f_buf)
        # self.logger.debug(f"EnvRunner(_read_batch): Done reading buffer. "
        #                   f"Final read index: {read_idx}.")

        return batches

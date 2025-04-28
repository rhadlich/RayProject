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
from ray.rllib.policy.sample_batch import SampleBatch

from multiprocessing import shared_memory
import struct
import subprocess
import sys


class SharedMemoryEnvRunner(EnvRunner):
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

        # weights communication properties go here
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
        self.STATE_ACTION_DIMS = self.config.env_config["ep_shm_properties"]["STATE_ACTION_DIMS"]
        self.episode_shm = shared_memory.SharedMemory(
            create=True,
            name=self.config.env_config["ep_shm_properties"].get("name", 'episodes'),
            size=self.TOTAL_SIZE
        )
        self.ep_buf = self.episode_shm.buf

        # Initialize the write and read indices in the episode buffer to 0
        struct.pack_into("<II", self.ep_buf, 0, 0, 0)  # write_idx=0, read_idx=0

        # Spawn minion subprocess. Decided to go with subprocess instead of threading because this way can split the
        # workload in the Raspberry Pi to be: 1 core Algorithm/Learner, 1 core EnvRunner, 1 core Minion (environment),
        # and the other core for the OS and any other support tasks.
        self._minion_proc = subprocess.Popen(
            [sys.executable, "Minion.py",
             "--weights_shm_name", self.weights_shm_name,
             "--episodes_shm_name", self.episode_shm.name,
             "--flag_shm_name", self.flag_shm_name]
        )

    def assert_healthy(self):
        """Checks that shared memory blocks have been created."""
        assert (
                self.weights_shm is not None and
                self.flag_shm is not None and
                self.episode_shm is not None and
                self._minion_proc.poll() is None
        ), "Shared memory blocks or minion process not alive."

    def sample(self, **kwargs):
        """
        Get new training batches.
        Returns an empty list if no new batches are available.
        """
        episodes = self._read_batch()

        # if no new batches were found
        if episodes is None:
            return []

        # if new batches were found
        num_env_steps = 0
        num_episodes_completed = 0
        for eps in episodes:
            assert eps.is_done
            self._done_episodes_for_metrics.append(eps)
            num_episodes_completed += 1
            num_env_steps += len(eps)

        # not sure why/if this is necessary but TCP example does it so couldn't hurt
        SingleAgentEnvRunner._increase_sampled_metrics(
            self, num_env_steps, num_episodes_completed
        )

        return episodes

    def get_metrics(self):
        # Compute per-episode metrics (only on already completed episodes).
        for eps in self._done_episodes_for_metrics:
            assert eps.is_done
            episode_length = len(eps)
            episode_return = eps.get_return()
            episode_duration_s = eps.get_duration_s()

            self._log_episode_metrics(
                episode_length, episode_return, episode_duration_s
            )

        # Now that we have logged everything, clear cache of done episodes.
        self._done_episodes_for_metrics.clear()

        # Return reduced metrics.
        return self.metrics.reduce()

    def stop(self):
        """Closes and unlinks the weights, flag, and episode shared memory blocks."""
        self._unlink_mem_blocks_if_necessary()

    def get_spaces(self):
        return {
            INPUT_ENV_SPACES: (self.config.observation_space, self.config.action_space),
            DEFAULT_MODULE_ID: (
                self.config.observation_space,
                self.config.action_space,
            ),
        }

    def get_state(self, state):
        return {
            COMPONENT_RL_MODULE: self.module.get_state(),
            WEIGHTS_SEQ_NO: self._weights_seq_no,
        }

    def set_state(self, state: StateDict) -> None:
        # Update the RLModule state.
        if COMPONENT_RL_MODULE in state:
            # A missing value for WEIGHTS_SEQ_NO or a value of 0 means: Force the
            # update.
            weights_seq_no = state.get(WEIGHTS_SEQ_NO, 0)

            # Only update the weights, if this is the first synchronization or
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

    # Helper functions
#   #------------------------------------------------------------------------#
    def _unlink_mem_blocks_if_necessary(self):
        if self.weights_shm:
            self.weights_shm.close()
            self.weights_shm.unlink()
        if self.flag_shm:
            self.flag_shm.close()
            self.flag_shm.unlink()
        if self.episode_shm:
            self.episode_shm.close()
            self.episode_shm.unlink()

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
    def get_indices(self):
        """Return (write_idx, read_idx)."""
        return struct.unpack_from("<II", self.ep_buf, 0)

    def set_indices(self, w, r):
        """Atomically update both indices (optionally split these)."""
        struct.pack_into("<II", self.ep_buf, 0, w, r)

    # I think this one goes in the minion
    def write_fragment(self, data: np.ndarray):
        """
        Append ONE rollout into the ring buffer.
        Drops the oldest slot if ring is full.

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
        assert data.shape == (self.ELEMENTS_PER_ROLLOUT,) and data.dtype == np.float32
        raw = data.tobytes()

        write_idx, read_idx = self.get_indices()
        slot_off = self.HEADER_SIZE + write_idx * self.SLOT_SIZE

        filled = struct.unpack_from("<H", self.ep_buf, slot_off)[0]  # uint16 count

        # Copy rollout into slot payload
        episode_off = slot_off + self.HEADER_SLOT_SIZE + filled * self.BYTES_PER_ROLLOUT
        self.ep_buf[episode_off: episode_off + self.BYTES_PER_ROLLOUT] = raw

        # Increment fill counter
        struct.pack_into("<H", self.ep_buf, slot_off, filled + 1)

        # If this is the last rollout that can be added to this slot -> move write_idx to next slot and populate the
        # first elements as the initial state (equal to last state of current write_idx).
        if filled == self.BATCH_SIZE:
            next_w = (write_idx + 1) % self.NUM_SLOTS
            if next_w == read_idx:  # ring full → drop oldest
                read_idx = (read_idx + 1) % self.NUM_SLOTS

            # get offset for next slot
            write_idx = next_w
            slot_off = self.HEADER_SIZE + write_idx * self.SLOT_SIZE

            # extract last state from current rollout (raw)
            state_off = len(raw) - self.STATE_ACTION_DIMS["state"] * int(self.BYTES_PER_ROLLOUT/self.ELEMENTS_PER_ROLLOUT)
            state_bytes = raw[state_off:]

            # add starting state to next slot
            initial_state_off = slot_off + self.HEADER_SLOT_SIZE
            self.ep_buf[initial_state_off: initial_state_off + len(state_bytes)] = state_bytes

            # reset the fill counter to include initial state
            struct.pack_into("<H", self.ep_buf, slot_off, len(state_bytes))

        # Commit updated indices
        self.set_indices(write_idx, read_idx)

    def _read_batch(self):
        """
        Pop the oldest COMPLETE 32-rollout batch from the ring.
        Returns list of SingleAgentEpisode objects or None if no complete slot is ready.
        """
        write_idx, read_idx = self.get_indices()
        # if any of these are true it means the last batch read is the most updated full batch
        if write_idx == read_idx or write_idx == read_idx+1 or (write_idx == 0 and read_idx == self.NUM_SLOTS-1):
            return None  # ring empty

        # In case the write_idx has wrapped around and there are more full batches available
        elif write_idx < read_idx:
            # write_idx-1 because the batch of write_idx is not full yet.
            num_batches = ((self.NUM_SLOTS-1) - read_idx) + (write_idx - 1)

        # regular case where there are batches available
        else:
            num_batches = write_idx - read_idx - 1

        batches = []
        for i in range(num_batches):
            slot_off = self.HEADER_SIZE + read_idx * self.SLOT_SIZE
            filled = struct.unpack_from("<H", self.ep_buf, slot_off)[0]
            assert filled == self.BATCH_SIZE

            # extract data from ring
            data_start = slot_off + self.HEADER_SLOT_SIZE
            payload = bytes(self.ep_buf[data_start: data_start + self.PAYLOAD_SIZE])

            # extract initial state from front of payload
            initial_state = payload[:self.STATE_ACTION_DIMS["state"] * int(self.BYTES_PER_ROLLOUT/self.ELEMENTS_PER_ROLLOUT)]

            # reshape data and add truncated and terminated fields
            flat = np.frombuffer(payload, dtype=np.float32).reshape(self.BATCH_SIZE, self.ELEMENTS_PER_ROLLOUT)
            terminateds = np.zeros(self.BATCH_SIZE, dtype=np.bool_)
            truncateds = np.zeros(self.BATCH_SIZE, dtype=np.bool_)
            truncateds[-1] = True

            # instantiate batch, format with SingleAgentEpisode
            batch = SingleAgentEpisode(
                observation_space=self.config.observation_space,
                action_space=self.config.action_space,
            )

            # add initial state
            batch.add_env_reset(np.frombuffer(initial_state, dtype=np.float32))

            # add one rollout at a time (i.e. one row in flat)
            for j in flat:
                batch.add_env_step(
                    observation=flat[j, 0:self.STATE_ACTION_DIMS["state"]],     # observation !AFTER! taking "action"
                    action=flat[j, self.STATE_ACTION_DIMS["state"]:self.STATE_ACTION_DIMS["action"]],
                    reward=flat[j, self.STATE_ACTION_DIMS["action"]:self.STATE_ACTION_DIMS["reward"]],
                    terminated=terminateds[j],
                    truncated=truncateds[j],
                )

            # append to list of batches to pass to learner
            batches.append(batch)

            # Advance read_idx
            read_idx = (read_idx + 1) % self.NUM_SLOTS

        # commit new indices
        self.set_indices(write_idx, read_idx)

        return batches

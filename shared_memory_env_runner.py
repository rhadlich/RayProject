from collections import defaultdict
from typing import DefaultDict, List, Optional

import numpy as np

from ray.rllib.core import (
    COMPONENT_RL_MODULE,
    DEFAULT_AGENT_ID,
    DEFAULT_MODULE_ID,
)
from ray.rllib.env import INPUT_ENV_SPACES
from ray.rllib.env.env_runner import EnvRunner
from ray.rllib.env.single_agent_env_runner import SingleAgentEnvRunner
from ray.rllib.env.single_agent_episode import SingleAgentEpisode
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
from ray.rllib.utils.typing import EpisodeID, StateDict

from multiprocessing import shared_memory
import Minion
import struct
import ray


# Helper functions
def get_indices(buf_arr) -> tuple:
    """Return (write_idx, read_idx)."""
    # return struct.unpack_from("<II", buf, 0)
    return buf_arr[:2].astype(np.int32)


def set_indices(buf_arr, w, r) -> shared_memory.SharedMemory.buf:
    """Atomically update both indices (optionally split these)."""
    # return struct.pack_into("<II", buf, 0, w, r)
    buf_arr[:2] = [w, r]


@ray.remote(num_cpus=1)
def _run_minion(policy_shm_name: str,
                flag_shm_name: str,
                ep_arr: np.ndarray,
                episode_shm_properties: dict
                ) -> None:
    """
    Thin wrapper that just calls minion to execute in a separate CPU core.
    """
    Minion.main(policy_shm_name, flag_shm_name, ep_arr, episode_shm_properties)


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
            policy_shm_name: name of the shared memory block that will contain model weights
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

        # Start shared memory block for episodes
        self.BATCH_SIZE = self.config.env_config["ep_shm_properties"]["BATCH_SIZE"]
        self.NUM_SLOTS = self.config.env_config["ep_shm_properties"]["NUM_SLOTS"]
        self.ELEMENTS_PER_ROLLOUT = self.config.env_config["ep_shm_properties"]["ELEMENTS_PER_ROLLOUT"]
        self.BYTES_PER_ROLLOUT = self.config.env_config["ep_shm_properties"]["BYTES_PER_ROLLOUT"]
        self.PAYLOAD_SIZE = self.config.env_config["ep_shm_properties"]["PAYLOAD_SIZE"]
        self.HEADER_SIZE = self.config.env_config["ep_shm_properties"]["HEADER_SIZE"]
        self.HEADER_SLOT_SIZE = self.config.env_config["ep_shm_properties"]["HEADER_SLOT_SIZE"]
        self.SLOT_SIZE = self.config.env_config["ep_shm_properties"]["SLOT_SIZE"]
        self.TOTAL_SIZE_BYTES = self.config.env_config["ep_shm_properties"]["TOTAL_SIZE_BYTES"]
        self.STATE_ACTION_DIMS = self.config.env_config["ep_shm_properties"]["STATE_ACTION_DIMS"]
        self.BYTES_PER_FLOAT = self.config.env_config["ep_shm_properties"]["BYTES_PER_FLOAT"]
        self.episode_shm = shared_memory.SharedMemory(
            create=True,
            name=self.config.env_config["ep_shm_properties"].get("name", 'episodes'),
            size=self.TOTAL_SIZE_BYTES
        )
        self.ep_buf = self.episode_shm.buf
        self.ep_arr = np.frombuffer(self.ep_buf, dtype=np.float32)
        self.episode_shm_name = self.episode_shm.name

        # Initialize the write and read indices in the episode buffer to 0
        self.ep_arr[:2] = 0
        # struct.pack_into("<II", self.ep_buf, 0, 0, 0)  # write_idx=0, read_idx=0

        # Spawn minion subprocess. Decided to go with ray task because it's convenient. This approach can split the
        # workload in the Raspberry Pi to be: 1 core Algorithm/Learner, 1 core EnvRunner, 1 core Minion (environment),
        # and the other core for the OS and any other support tasks.
        self._minion_ref = _run_minion.remote(
            self.policy_shm_name,
            self.flag_shm_name,
            self.ep_arr,
            self.config.env_config["ep_shm_properties"],
        )

    def assert_healthy(self):
        """Checks that shared memory blocks and minion process have been created."""
        minion_running, _ = ray.wait([self._minion_ref], timeout=0)
        if not minion_running:
            minion_running = None

        # in the first iteration, get shm references
        if self.policy_shm == False:
            self._get_shm_references()

        assert (
                self.policy_shm is not None and
                self.flag_shm is not None and
                self.episode_shm is not None and
                minion_running is not None
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
    def _get_shm_references(self):
        # get reference to policy shared memory weights
        self.policy_shm = shared_memory.SharedMemory(name=self.policy_shm_name, create=False)
        self.flag_shm = shared_memory.SharedMemory(name=self.flag_shm_name, create=False)

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
        write_idx, read_idx = get_indices(self.ep_arr)
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
            # assert that the batch/slot is full
            slot_off = self.HEADER_SIZE + read_idx * self.SLOT_SIZE
            filled = int(self.ep_arr[slot_off])
            assert filled == self.BATCH_SIZE

            # extract data from ring
            data_start = slot_off + self.HEADER_SLOT_SIZE
            payload = np.copy(self.ep_arr[data_start: data_start + self.PAYLOAD_SIZE])  # extract data from array

            # extract initial state from front of payload and remove it so that shape is consistent for next step
            initial_state = payload[:self.STATE_ACTION_DIMS["state"]]
            payload = np.delete(payload, np.arange(self.STATE_ACTION_DIMS["action"]))

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
            batch.add_env_reset(initial_state)

            # add one rollout at a time (i.e. one row in flat)
            for j in payload:
                batch.add_env_step(
                    observation=payload[j, 0:self.STATE_ACTION_DIMS["state"]],     # observation !AFTER! taking "action"
                    action=payload[j, self.STATE_ACTION_DIMS["state"]:self.STATE_ACTION_DIMS["action"]],
                    reward=payload[j, self.STATE_ACTION_DIMS["action"]:self.STATE_ACTION_DIMS["reward"]],
                    terminated=terminateds[j],
                    truncated=truncateds[j],
                )

            # append to list of batches to pass to learner
            batches.append(batch)

            # Advance read_idx
            read_idx = (read_idx + 1) % self.NUM_SLOTS

        # commit new indices
        self.ep_buf = set_indices(self.ep_buf, write_idx, read_idx)

        return batches



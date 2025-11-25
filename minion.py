from multiprocessing import shared_memory
from typing import Optional, Union

import numpy as np
import struct
import onnxruntime as ort
import time
import gzip
import os
import torch

from ray.rllib.utils.numpy import softmax
import gymnasium as gym
from gymCustom import reward_fn, EngineEnvDiscrete, EngineEnvContinuous

from utils import ActionAdapter

from ray.rllib.env import INPUT_ENV_SPACES
from ray.rllib.core import DEFAULT_MODULE_ID

import logging
import logging_setup
from pprint import pformat

# Try to import zmq, but make it optional
try:
    import zmq
    zmq_available = True
except ImportError:
    zmq_available = False
    zmq = None


def get_indices(buf_arr, f_buf, *, logger=None) -> tuple:
    """Return (write_idx, read_idx)."""
    f_buf[3] = 1  # lock episode buffer
    indices = buf_arr[:2].astype(np.int32)
    return indices


def set_indices(buf_arr: np.ndarray,
                idx: int,
                mode: str,
                f_buf: shared_memory.SharedMemory.buf) -> None:
    """Atomically update either write or read index."""
    # return struct.pack_into("<II", buf, 0, w, r)
    if mode == 'w':
        buf_arr[0] = idx
    elif mode == 'r':
        buf_arr[1] = idx
    else:
        raise ValueError(f'Unknown mode {mode}')
    f_buf[3] = 0  # unlock episode buffer


def flatten_obs_onehot(obs, imep_space, mprr_space) -> torch.Tensor:
    """
    Function that takes in an observation from the environment and flattens to
    the shape required by the ort session.

    The ort session expects a one-hot encoding of the observation. RLlib converts
    observation spaces like the ones here to one-hot representation. There are
    built-in tools to handle this, but decided to with this implementation
    because it is faster and have more control. Also, the built-in tools weren't
    working.
    """
    # imep, mprr = obs["state"]
    tgt = obs

    # imep_idx = np.argmin(np.abs(imep - imep_space))
    # mprr_idx = np.argmin(np.abs(mprr - mprr_space))
    tgt_imep_idx = np.argmin(np.abs(tgt - imep_space))

    # imep_cur = np.eye(len(imep_space), dtype=np.float32)[imep_idx]  # (n_imep,)
    # mprr_cur = np.eye(len(mprr_space), dtype=np.float32)[mprr_idx]  # (n_mprr,)
    tgt_imep = np.eye(len(imep_space), dtype=np.float32)[tgt_imep_idx]  # (n_imep,)

    # return torch.tensor(np.concatenate([imep_cur, mprr_cur, imep_tgt]))
    return torch.tensor(tgt_imep, dtype=torch.float32)


def _flatten_obs_array(obs) -> np.ndarray:
    """
    Function that flattens the observation so that it can be stored in the
    shared memory buffer.
    """
    # return np.append(obs["state"], obs["target"]).astype(np.float32)
    return np.expand_dims(obs, 0).astype(np.float32)


class Minion:
    def __init__(
            self,
            policy_shm_name: str,
            flag_shm_name: str,
            ep_shm_name: str,
            config,
    ):
        # create logger
        self.logger = logging.getLogger("MyRLApp.Minion")
        self.logger.info(f"Minion, PID={os.getpid()}")
        self.logger.debug("Minion: Started __init__()")

        # add attributes to object
        # self.policy_shm_name = policy_shm_name
        # self.flag_shm_name = flag_shm_name
        # self.ep_shm_name = ep_shm_name
        self.config = config
        self.policy_shm_name = self.config.env_config['policy_shm_name']
        self.flag_shm_name = self.config.env_config['flag_shm_name']
        self.episode_shm_properties = self.config.env_config["ep_shm_properties"]
        self.ep_shm_name = self.episode_shm_properties['name']

        # set parameters for training and evaluation
        self.reset_env_each_n_batches = False
        self.n_batches_for_env_reset = 50

        # connect to shared memory blocks
        self.f_shm = shared_memory.SharedMemory(name=self.flag_shm_name, create=False)  # this one has to be first
        self.f_buf = self.f_shm.buf
        self.ep_shm = shared_memory.SharedMemory(name=self.ep_shm_name, create=False)
        self.ep_buf = self.ep_shm.buf
        self.ep_arr = np.ndarray(shape=(self.episode_shm_properties["TOTAL_SIZE"],),
                                 dtype=np.float32,
                                 buffer=self.ep_buf,
                                 )
        while self.f_buf[0] == 1:  # wait until policy shared memory block has been created
            time.sleep(0.01)
        # connect to policy shared memory block and get buffer pointer
        self.p_shm = shared_memory.SharedMemory(name=policy_shm_name, create=False)
        self.p_buf = self.p_shm.buf

        self.logger.debug(f"Minion: ep_arr shape -> {self.ep_arr.shape}")
        self.logger.debug("Minion: connected to all memory blocks")

        self.logger.debug("Minion: Getting initial network weights")
        self.ort_session = None
        # get initial network weights
        while self.f_buf[1] == 0:  # wait until weights-available flag is set to true
            time.sleep(0.01)
        self.ort_session, self.input_names, self.output_names = self._get_ort_session()
        self.logger.debug(f"Minion: input_names: {self.input_names}, output_names: {self.output_names}")
        self.f_buf[1] = 0  # change new-weights-available flag to false

        self.logger.debug("Minion: Initialized ORT session")

        # initialize environment (gym.Env or socket to LabVIEW)
        env_type = self.config.env_config['env_type']
        if env_type == 'continuous':
            self.env = EngineEnvContinuous(reward=reward_fn)
        elif env_type == 'discrete':
            self.env = EngineEnvDiscrete(reward=reward_fn)
        else:
            raise NotImplementedError(f"Environment type not supported or not provided.")

        if isinstance(self.env.observation_space, gym.spaces.Discrete):
            self.obs_is_discrete = True

            # extract action and observation spaces dimensions
            # self.sizes = [sp.n for sp in self.env.action_space]
            # self.cuts = np.cumsum(self.sizes)[:-1]
            self.len_imep = len(self.env.imep_space)
            self.len_mprr = len(self.env.mprr_space)
        elif isinstance(self.env.observation_space, gym.spaces.Box):
            self.obs_is_discrete = False
        else:
            raise NotImplementedError(f'Unknown observation_space {self.env.observation_space}')

        self.logger.debug(f"Minion: obs_is_discrete={self.obs_is_discrete}")

        # get random reference observation to check ort outputs and make sure weights change
        if self.obs_is_discrete:
            obs_shape = self.len_imep
        else:
            obs_shape = self.episode_shm_properties["STATE_ACTION_DIMS"]["state"]
        self.ref_obs = np.random.randn(32, obs_shape).astype(np.float32)
        self.old_policy_output = None

        # initialize action adapter, build module and extract action_dist_cls to sample actions properly
        spaces = {
            INPUT_ENV_SPACES: (self.config.observation_space, self.config.action_space),
            DEFAULT_MODULE_ID: (
                self.config.observation_space,
                self.config.action_space,
            ),
        }
        module_spec = self.config.get_rl_module_spec(
            spaces=spaces, inference_only=True
        )
        module = module_spec.build()
        self.action_dist_cls = module.get_inference_action_dist_cls()
        self.action_adapter = ActionAdapter(self.env.action_space, action_dist_cls=self.action_dist_cls)

        self.logger.debug("Minion: Initialized ENV.")

        # set up data broadcasting to GUI (optional)
        self.pub = None
        self.zmq_ctx = None
        enable_zmq = self.config.env_config.get("enable_zmq", True)
        if enable_zmq and zmq_available and zmq is not None:
            try:
                self.zmq_ctx = zmq.Context()
                self.pub = self.zmq_ctx.socket(zmq.PUB)
                self.pub.bind("ipc:///tmp/engine.ipc")
                self.logger.info("ZMQ publisher initialized for GUI communication")
            except Exception as e:
                self.logger.warning(f"Failed to initialize ZMQ publisher: {e}. Continuing without ZMQ.")
                self.pub = None
                self.zmq_ctx = None
        elif enable_zmq and not zmq_available:
            self.logger.warning("ZMQ requested but not available (zmq not installed). Continuing without ZMQ.")
        else:
            self.logger.debug("ZMQ disabled via config")

        # start count
        self.batch_count = 0
        self.last_obs = None

        self.logger.debug("Minion: Done with __init__().")

    def _ort_session_run(self, session, obs):
        net_out = session.run(
            self.output_names,
            {self.input_names[0]: obs},
        )
        return net_out

    def _weights_changed(self, new_sess, atol=1e-5):
        self.logger.debug("Minion: Called _weights_changed")

        out_new = []
        for obs in self.ref_obs:
            out_new.append(self._ort_session_run(new_sess, np.array([obs], np.float32))[0])
        out_new = np.array(out_new)

        if self.old_policy_output is None:
            self.old_policy_output = out_new
            return True

        diff = np.sum(np.abs(self.old_policy_output - out_new))
        self.logger.debug(f"Δpolicy={diff:.4e}")

        msg = {
            "topic": "policy",
            "delta in minion": float(diff),
        }
        # self.logger.debug(f"Minion (train_and_eval_sequence): eval msg: {msg}.")
        if self.pub is not None:
            self.pub.send_json(msg)

        self.old_policy_output = out_new

        self.logger.debug("Minion: Finished _weights_changed")

        return diff > atol

    def _get_ort_session(self, ):
        self.logger.debug("Minion: Called _get_ort_session")
        # get length of ort_compressed from header
        _len_ort = struct.unpack_from("<I", self.p_buf, 0)
        header_offset = 4  # number of bytes in the int

        ort_raw = self.p_buf[header_offset:header_offset + _len_ort[0]].tobytes()  # this is the actual ort model bytes

        ort_session = ort.InferenceSession(
            ort_raw,
            providers=[("CoreMLExecutionProvider",
                        {"ModelFormat": "MLProgram", "MLComputeUnits": "ALL", "RequireStaticInputShapes": "1"}
                        ),
                       "CPUExecutionProvider", ]
        )
        output_names = [o.name for o in ort_session.get_outputs()]

        input_names = [i.name for i in ort_session.get_inputs()]

        return ort_session, input_names, output_names

    def try_update_ort_weights(self) -> bool:
        # update policy network weights if available and not being written to
        if self.f_buf[1] == 1 and self.f_buf[0] == 0:
            self.logger.debug("Minion: Updating ort session weights...")
            self.f_buf[0] = 1  # set lock flag to locked
            tic = time.time()
            self.ort_session, self.input_names, self.output_names = self._get_ort_session()  # get ort session with new weights
            toc = time.time()
            self.logger.debug(f"Minion: Time to update ort weights is {(toc - tic)*1000:0.4f}ms")

            # check if policy changed
            try:
                policy_changed = self._weights_changed(self.ort_session)
                self.logger.debug(f"Minion: Policy weights changed? {policy_changed}")
            except Exception as e:
                self.logger.debug(f"Minion: Could not check policy weights due to error {e}")
                raise RuntimeError from e

            self.f_buf[0] = 0  # set lock flag to unlocked
            self.f_buf[1] = 0  # reset weights-available flag to 0 (false, i.e. no new weights)
            self.logger.debug("Minion: Ort weights updated.")
            return True
        else:
            return False

    def write_fragment(
            self,
            data: np.ndarray,
            is_initial_state: Optional[bool] = False,
    ) -> np.ndarray:
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
        │   • filled_count  (float32)      ← how many rollouts │
        │                        (# of rollouts,not byte count)│
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
        if is_initial_state:
            assert data.shape == (
                self.episode_shm_properties["STATE_ACTION_DIMS"]['state'],) and data.dtype == np.float32
        else:
            assert data.shape == (self.episode_shm_properties["ELEMENTS_PER_ROLLOUT"],) and data.dtype == np.float32

        # wait until the buffer is unlocked to read indices, then read and lock (locking happens in get_indices)
        while True:
            if self.f_buf[3] == 0:
                write_idx, read_idx = get_indices(self.ep_arr, self.f_buf, logger=self.logger)
                # self.logger.debug(f"Minion (write_fragment): Started writing fragment. "
                #              f"Identified writing and reading indices: {write_idx}, {read_idx}.")
                break
            else:
                time.sleep(0.0001)

        slot_off = self.episode_shm_properties["HEADER_SIZE"] + write_idx * self.episode_shm_properties["SLOT_SIZE"]

        # self.logger.debug(f"Minion (write_fragment): identified writing and reading indices: {write_idx}, {read_idx}.")
        # self.logger.debug(f"Minion (write_fragment): ELEMENTS_PER_ROLLOUT: {episode_shm_properties['ELEMENTS_PER_ROLLOUT']}.")
        # self.logger.debug(f"Minion (write_fragment): data shape: {data.shape}.")

        # only run this the very first time the episode buffer is started,
        # for other iterations the function already handles adding the initial state
        if is_initial_state:
            initial_state_off = slot_off + self.episode_shm_properties["HEADER_SLOT_SIZE"]
            self.ep_arr[initial_state_off: initial_state_off + len(data)] = data
            set_indices(self.ep_arr, write_idx, 'w', self.f_buf)  # to unlock episode buffer
            self.logger.debug(f"Minion (write_fragment): Done writing initial state."
                              f"Final writing index: {write_idx}.")
            return self.ep_arr

        # get how many rollouts have been filled in the current episode
        filled = int(self.ep_arr[slot_off])

        # self.logger.debug(f"Minion (write_fragment): writing to slot {filled}.")

        # Copy rollout into slot payload
        episode_off = slot_off + self.episode_shm_properties["HEADER_SLOT_SIZE"] + filled * self.episode_shm_properties[
            "ELEMENTS_PER_ROLLOUT"]
        # Add initial state to offset
        episode_off += self.episode_shm_properties["STATE_ACTION_DIMS"]['state']
        self.ep_arr[episode_off: episode_off + self.episode_shm_properties["ELEMENTS_PER_ROLLOUT"]] = data

        # Increment fill counter
        filled += 1
        self.ep_arr[slot_off] = filled

        # If this is the last rollout that can be added to this slot -> move write_idx to next slot and populate the
        # first elements as the initial state (equal to last state of current write_idx).
        #
        # Alternatively, can pass the env and set reset_env_each_batch flag to True. This will make initial observation
        # of each batch equal to the output of reset.
        if filled == self.episode_shm_properties["BATCH_SIZE"]:
            next_w = (write_idx + 1) % self.episode_shm_properties["NUM_SLOTS"]
            if next_w == read_idx:  # ring full → drop oldest
                self.logger.debug("Minion: ring buffer got filled, writing episodes faster than algorithm can read.")
                read_idx = (read_idx + 1) % self.episode_shm_properties["NUM_SLOTS"]

            # get offset for next slot
            write_idx = next_w
            slot_off = self.episode_shm_properties["HEADER_SIZE"] + write_idx * self.episode_shm_properties["SLOT_SIZE"]

            if self.reset_env_each_n_batches and not self.batch_count % self.n_batches_for_env_reset:
                # get next state from env.reset()
                self.logger.debug(f"Minion: resetting env in batch {self.batch_count}")
                obs, info = self.env.reset()
                state = _flatten_obs_array(obs)
            else:
                # extract last state from current rollout to use as initial observation for buffer slot
                state_off = (self.episode_shm_properties["STATE_ACTION_DIMS"]['action'] +
                             self.episode_shm_properties["STATE_ACTION_DIMS"]['reward'])
                state = data[state_off:state_off + self.episode_shm_properties["STATE_ACTION_DIMS"]['state']]

            # add initial state to next slot
            initial_state_off = slot_off + self.episode_shm_properties["HEADER_SLOT_SIZE"]
            self.ep_arr[initial_state_off: initial_state_off + len(state)] = state

            # reset the fill counter
            self.ep_arr[slot_off] = 0

            # increment batch count
            self.batch_count += 1

        # Commit updated indices and unlock episode buffer (unlocking happens inside set_indices)
        set_indices(self.ep_arr, write_idx, 'w', self.f_buf)
        # logger.debug(f"Minion (write_fragment): Done writing fragment."
        #              f"Final writing index: {write_idx}.")

        return self.ep_arr

    def collect_rollouts(
            self,
            n_rollouts: int = 1,
            initial_obs: Optional[np.ndarray] = None,
            deterministic: Optional[bool] = False,
    ) -> Union[dict, list]:
        """
        function to collect a set number of rollouts
        """

        if initial_obs is None:
            obs, info = self.env.reset()
        else:
            obs = initial_obs

        if n_rollouts > 1:
            rollouts = {
                "obs": [],
                "actions": [],
                "rewards": [],
                "terminateds": [],
                "truncateds": [],
                "action_dist_inputs": [],
                "action_logps": [],
                "info": []
            }  # dict to store the rollouts

        for i in range(n_rollouts):

            # self.logger.debug("Minion: in loop to collect rollouts")

            if self.obs_is_discrete:
                # one-hot encode observation
                obs_for_model = np.array([flatten_obs_onehot(obs, self.env.imep_space, self.env.mprr_space)],
                                         np.float32)
            else:
                obs_for_model = np.array([obs], np.float32)

            if obs_for_model.ndim == 1:
                obs_for_model = np.expand_dims(obs_for_model, axis=0)

            # self.logger.debug(f"Minion: obs_for_model: {obs_for_model}")

            tic2 = time.time()
            try:
                # inference pass through actor network
                # first [0] -> selects "output". second [0] -> selects 0th batch
                net_out = self._ort_session_run(self.ort_session, obs_for_model)[0][0]
            except Exception as e:
                raise RuntimeError(f"Could not perform action inference due to error {e}")

            toc2 = time.time()
            # self.logger.debug(f"Minion time for inference only is {(toc2 - tic2)*1000:0.4f}ms")

            # self.logger.debug(f"Minion: performed action inference, net_out={net_out}, type={type(net_out)}")

            action_sampled, logp, dist_inputs = self.action_adapter.sample_from_policy(net_out,
                                                                                       deterministic=deterministic)

            # self.logger.debug(
            #     f"Minion: sampled action: action_raw={action_sampled}, logp={logp}, dist_inputs={dist_inputs}")

            if self.action_adapter.mode == "continuous":
                action_for_env = np.concatenate(([550, ], self.action_adapter.get_action_in_env_range(action_sampled)),
                                                dtype=np.float32)
            else:
                action_for_env = np.concatenate(([1, ], action_sampled), dtype=np.int32)

            # self.logger.debug(f"Minion: made action vector: action_for_env={action_for_env}")

            obs, reward, terminated, truncated, info = self.env.step(action_for_env)

            # self.logger.debug(f"Minion: sampled new rollout. obs={obs}")

            if n_rollouts == 1:
                return [obs, action_sampled, reward, terminated, truncated, logp, net_out, info]
            else:
                rollouts["obs"].append(obs)
                rollouts["actions"].append(action_sampled)
                rollouts["rewards"].append(reward)
                rollouts["terminateds"].append(terminated)
                rollouts["truncateds"].append(truncated)
                rollouts["action_logps"].append(logp)
                rollouts["action_dist_inputs"].append(net_out)
                rollouts["info"].append(info)

        return rollouts

    def train_and_eval_sequence(
            self,
            train_batches: int = 1,
            eval_rollouts: int = 1,
    ):

        if not self.last_obs:
            obs, info = self.env.reset()
            self.write_fragment(_flatten_obs_array(obs), is_initial_state=True)
        else:
            obs = self.last_obs

        for i in range(int(train_batches * self.episode_shm_properties["BATCH_SIZE"])):
            # self.logger.debug(f"Minion: in loop to collect batch, iter={i}")
            tic = time.time()
            obs, action, reward, terminated, truncated, logp, net_out, info = (
                self.collect_rollouts(initial_obs=obs))
            toc = time.time()
            # self.logger.debug(f"Minion: time to collect full rollout is {(toc - tic)*1000:0.4f}ms")


            # self.logger.debug(f"Minion (train_and_eval_sequence): received new rollout.")
            # self.logger.debug(f"Minion (train_and_eval_sequence): obs: {obs}.")
            # self.logger.debug(f"Minion (train_and_eval_sequence): action: {action}.")
            # self.logger.debug(f"Minion (train_and_eval_sequence): reward: {reward}.")
            # self.logger.debug(f"Minion (train_and_eval_sequence): logp: {logp}.")
            # self.logger.debug(f"Minion (train_and_eval_sequence): info: {info}.")

            obs_flat = _flatten_obs_array(obs)  # flatten to shape needed by memory buffer
            current_packet = np.concatenate((
                action,
                np.array([reward], dtype=np.float32),
                obs_flat,
                np.array([logp], dtype=np.float32),
                net_out.astype(np.float32),
            )).astype(np.float32)

            # self.logger.debug(f"Minion (train_and_eval_sequence): built packet.")

            # write it into the buffer
            self.write_fragment(current_packet)

            # self.logger.debug(f"Minion (train_and_eval_sequence): wrote to buffer.")

            # send training results to be logged in the GUI
            msg = {
                "topic": "engine",
                "current imep": float(info["current imep"]),
                "mprr": float(info["mprr"]),
                "target": float(obs)
            }
            # self.logger.debug(f"Minion (train_and_eval_sequence): msg: {msg}.")
            if self.pub is not None:
                try:
                    self.pub.send_json(msg)
                except Exception as e:
                    self.logger.debug(f"Minion (train_and_eval_sequence): {e}")

            # self.logger.debug(f"Minion (train_and_eval_sequence): sent to GUI.")

        # set last observation
        self.last_obs = obs

        for i in range(eval_rollouts):
            obs, action, _, _, _, _, _, info = (
                self.collect_rollouts(initial_obs=obs, deterministic=True))

            # send evaluation results to be logged in the GUI
            msg = {
                "topic": "evaluation",
                "current imep": float(info["current imep"]),
                "mprr": float(info["mprr"]),
                "target": float(obs)
            }
            # self.logger.debug(f"Minion (train_and_eval_sequence): eval msg: {msg}.")
            if self.pub is not None:
                self.pub.send_json(msg)


def main(policy_shm_name: str,
         flag_shm_name: str,
         ep_shm_name: str,
         episode_shm_properties: dict,
         ):
    """
    Function that runs minion to interact with the environment. Structure is:

    │ connect to shared memory blocks
    │ load initial policy network weights
    │ initialize environment (gym.Env or LabVIEW socket)
    │ Get initial network weights
    │ initialize episode collection buckets

    │ while True
    │ │ receive state (and maybe reward) from environment
    │ │ perform policy inference to sample actions
    │ │ send action to environment
    │ │ log state, action, reward into buckets
    │ │ if batch size or episode length reached
    │ │ │ write episode data to shared memory block
    │ │ │ clear buckets


    """

    actor = Minion(
        policy_shm_name,
        flag_shm_name,
        ep_shm_name,
        episode_shm_properties
    )

    timesteps = 0
    weight_updates = 0
    # store_rollout = True

    try:
        while True:

            weights_updated = actor.try_update_ort_weights()
            if weights_updated:
                actor.logger.debug(f"Minion: Update number -> {weight_updates}.")
                weight_updates += 1

            # perform train and eval routine
            actor.train_and_eval_sequence(
                train_batches=1,
                eval_rollouts=1,
            )

            # set minion rollout flag to true to enable the algo.train() calls
            actor.f_buf[2] = 1

            # logger.debug(f"Minion: Done with iteration {timesteps}")

            # if environment is the physical engine, wait for new state update and reward (simulated with a sleep)
            time.sleep(0.01)

            timesteps += 1

    except KeyboardInterrupt:
        # close socket connection
        del actor.ep_arr
        actor.logger.debug("Program interrupted")

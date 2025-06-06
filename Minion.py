from multiprocessing import shared_memory
import numpy as np
import struct
import onnxruntime as ort
import time
import gzip
import os
import torch

from ray.rllib.utils.numpy import softmax
import gymnasium as gym
from gymCustom import reward_fn, EngineEnv

import logging
import logging_setup


def get_indices(buf_arr, f_buf, *, logger=None) -> tuple:
    """Return (write_idx, read_idx)."""
    f_buf[3] = 1    # lock episode buffer
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
    f_buf[3] = 0    # unlock episode buffer


def _get_ort_session(buf: shared_memory.SharedMemory.buf, logger):
    logger.debug("Minion: Called _get_ort_session")
    # get length of ort_compressed from header
    _len_ort = struct.unpack_from("<I", buf, 0)
    header_offset = 4  # number of bytes in the int

    ort_raw = buf[header_offset:header_offset + _len_ort[0]].tobytes()  # this is the actual ort model (bytes)

    try:
        ort_session = ort.InferenceSession(
            ort_raw,
            providers=[("CoreMLExecutionProvider",
                        {"ModelFormat": "MLProgram", "MLComputeUnits": "ALL", "RequireStaticInputShapes": "1"}
                        ),
                       "CPUExecutionProvider", ]
        )
        output_names = [o.name for o in ort_session.get_outputs()]
    except Exception as e:
        logger.debug(f"Minion: Could not start ort session: {e}")

    return ort_session, output_names


def _flatten_obs_onehot(obs, imep_space, mprr_space) -> torch.Tensor:
    """
    Function that takes in an observation from the environment and flattens to
    the shape required by the ort session.

    The ort session expects a one-hot encoding of the observation. RLlib converts
    observation spaces like the ones here to one-hot representation. There are
    built-in tools to handle this, but decided to with this implementation
    because it is faster and have more control. Also, the built-in tools weren't
    working.
    """
    imep, mprr = obs["state"]
    tgt = obs["target"]

    imep_idx = np.argmin(np.abs(imep - imep_space))
    mprr_idx = np.argmin(np.abs(mprr - mprr_space))
    tgt_idx = np.argmin(np.abs(tgt - mprr_space))

    imep_cur = np.eye(len(imep_space), dtype=np.float32)[imep_idx]  # (n_imep,)
    mprr_cur = np.eye(len(mprr_space), dtype=np.float32)[mprr_idx]  # (n_mprr,)
    imep_tgt = np.eye(len(imep_space), dtype=np.float32)[tgt_idx]  # (n_imep,)

    return torch.tensor(np.concatenate([imep_cur, mprr_cur, imep_tgt]))


def _flatten_obs_array(obs) -> np.ndarray:
    """
    Function that flattens the observation so that it can be stored in the
    shared memory buffer.
    """
    return np.append(obs["state"], obs["target"]).astype(np.float32)


def write_fragment(data: np.ndarray,
                   buf_arr: np.ndarray,
                   f_buf: shared_memory.SharedMemory.buf,
                   episode_shm_properties: dict,
                   logger: logging.Logger,
                   *,
                   is_initial_state: bool = False,
                   env: gym.Env = None,
                   reset_env_each_batch: bool = False,
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
        assert data.shape == (episode_shm_properties["STATE_ACTION_DIMS"]['state'],) and data.dtype == np.float32
    else:
        assert data.shape == (episode_shm_properties["ELEMENTS_PER_ROLLOUT"],) and data.dtype == np.float32

    # wait until the buffer is unlocked to read indices, then read and lock (locking happens in get_indices)
    while True:
        if f_buf[3] == 0:
            write_idx, read_idx = get_indices(buf_arr, f_buf, logger=logger)
            # logger.debug(f"Minion (write_fragment): Started writing fragment. "
            #              f"Identified writing and reading indices: {write_idx}, {read_idx}.")
            break
        else:
            time.sleep(0.0001)

    slot_off = episode_shm_properties["HEADER_SIZE"] + write_idx * episode_shm_properties["SLOT_SIZE"]

    # logger.debug(f"Minion (write_fragment): identified writing and reading indices: {write_idx}, {read_idx}.")
    # logger.debug(f"Minion (write_fragment): ELEMENTS_PER_ROLLOUT: {episode_shm_properties['ELEMENTS_PER_ROLLOUT']}.")
    # logger.debug(f"Minion (write_fragment): data shape: {data.shape}.")

    # only run this the very first time the episode buffer is started,
    # for other iterations the function already handles adding the initial state
    if is_initial_state:
        initial_state_off = slot_off + episode_shm_properties["HEADER_SLOT_SIZE"]
        buf_arr[initial_state_off: initial_state_off + len(data)] = data
        set_indices(buf_arr, write_idx, 'w', f_buf)    # to unlock episode buffer
        # logger.debug(f"Minion (write_fragment): Done writing initial state."
        #              f"Final writing index: {write_idx}.")
        return buf_arr

    # get how many rollouts have been filled in the current episode
    filled = int(buf_arr[slot_off])

    # logger.debug(f"Minion (write_fragment): writing to slot {filled}.")

    # Copy rollout into slot payload
    episode_off = slot_off + episode_shm_properties["HEADER_SLOT_SIZE"] + filled * episode_shm_properties[
        "ELEMENTS_PER_ROLLOUT"]
    # Add initial state to offset
    episode_off += episode_shm_properties["STATE_ACTION_DIMS"]['state']
    buf_arr[episode_off: episode_off + episode_shm_properties["ELEMENTS_PER_ROLLOUT"]] = data

    # Increment fill counter
    filled += 1
    buf_arr[slot_off] = filled

    # If this is the last rollout that can be added to this slot -> move write_idx to next slot and populate the
    # first elements as the initial state (equal to last state of current write_idx).
    #
    # Alternatively, can pass the env and set reset_env_each_batch flag to True. This will make initial observation
    # of each batch equal to the output of reset.
    if filled == episode_shm_properties["BATCH_SIZE"]:
        next_w = (write_idx + 1) % episode_shm_properties["NUM_SLOTS"]
        if next_w == read_idx:  # ring full → drop oldest
            read_idx = (read_idx + 1) % episode_shm_properties["NUM_SLOTS"]

        # get offset for next slot
        write_idx = next_w
        slot_off = episode_shm_properties["HEADER_SIZE"] + write_idx * episode_shm_properties["SLOT_SIZE"]

        if env is not None and reset_env_each_batch:
            # get next state from env.reset()
            obs, info = env.reset()
            state = _flatten_obs_array(obs)
        else:
            # extract last state from current rollout
            state_off = len(data) - episode_shm_properties["STATE_ACTION_DIMS"]['state']
            state = data[state_off:]

        # add starting state to next slot
        initial_state_off = slot_off + episode_shm_properties["HEADER_SLOT_SIZE"]
        buf_arr[initial_state_off: initial_state_off + len(state)] = state

        # reset the fill counter
        buf_arr[slot_off] = 0

    # Commit updated indices and unlock episode buffer (unlocking happens inside set_indices)
    set_indices(buf_arr, write_idx, 'w', f_buf)
    # logger.debug(f"Minion (write_fragment): Done writing fragment."
    #              f"Final writing index: {write_idx}.")

    return buf_arr


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

    logger = logging.getLogger("MyRLApp.Minion")
    logger.info(f"Minion, PID={os.getpid()}")

    logger.debug("Minion: Started main()")

    # connect to shared memory
    f_shm = shared_memory.SharedMemory(name=flag_shm_name, create=False)  # this one has to be first
    f_buf = f_shm.buf
    ep_shm = shared_memory.SharedMemory(name=ep_shm_name, create=False)
    ep_buf = ep_shm.buf
    ep_arr = np.ndarray(shape=(episode_shm_properties["TOTAL_SIZE"],),
                        dtype=np.float32,
                        buffer=ep_buf,
                        )
    logger.debug(f"Minion: ep_arr shape -> {ep_arr.shape}")

    logger.debug("Minion: connected to flag memory block")

    while f_buf[0] == 1:  # wait until other shared memory blocks have been created
        time.sleep(0.01)

    p_shm = shared_memory.SharedMemory(name=policy_shm_name, create=False)

    logger.debug("Minion: Connected to policy memory block")

    # get buffers
    p_buf = p_shm.buf

    logger.debug("Minion: Getting initial network weights")
    # get initial network weights
    while f_buf[1] == 0:  # wait until weights-available flag is set to true
        time.sleep(0.01)
    ort_session, output_names = _get_ort_session(p_buf, logger=logger)
    f_buf[1] = 0  # change new-weights-available flag to false

    logger.debug("Minion: Initialized ORT session")

    # initialize environment (gym.Env or socket to LabVIEW)
    # get env.reset() also, meaning system's initial state observation
    env = EngineEnv(reward=reward_fn)

    imep_space = env.imep_space
    mprr_space = env.mprr_space

    obs, info = env.reset()

    # extract action and observation spaces dimensions
    sizes = [sp.n for sp in env.action_space]
    cuts = np.cumsum(sizes)[:-1]
    len_imep = env.observation_space["state"].nvec[0]
    len_mprr = env.observation_space["state"].nvec[1]

    # flatten initial observation to one-hot and np array representations
    obs_onehot = _flatten_obs_onehot(obs, imep_space, mprr_space)
    obs_flat = _flatten_obs_array(obs)

    logger.debug("Minion: Initialized ENV.")

    try:
        # write initial state to episode buffer
        write_fragment(obs_flat,
                       ep_arr,
                       f_buf,
                       episode_shm_properties,
                       is_initial_state=True,
                       logger=logger, )
    except Exception as e:
        logger.debug(f"Minion: could not write fragment due to error: {e}")

    timesteps = 0
    weight_updates = 0

    try:
        while True:
            timesteps += 1

            # perform action inference using the ort model (sample probabilities)
            logits = ort_session.run(
                output_names,
                {"onnx::Gemm_0": np.array([obs_onehot], np.float32)},
            )[0][0]  # first [0] -> selects "output". second [0] -> selects 0th batch

            # stochastic sample of actions (for discrete action space only)
            logits_soi, logits_d = np.split(logits, cuts)
            soi_probs = softmax(logits_soi)
            inj_d_probs = softmax(logits_d)
            idx_soi = int(np.random.choice(sizes[0], p=soi_probs))
            idx_inj_d = int(np.random.choice(sizes[1], p=inj_d_probs))
            logp = float(
                np.log(soi_probs[idx_soi]) +
                np.log(inj_d_probs[idx_inj_d])
            )       # joint log-probability of multi-branch action spa

            # send action to environment (and in the case of gym.Env collect new observation and reward)
            action = np.array([1, idx_soi, idx_inj_d], dtype=np.int32)    # 1 is for inj_p, to be kept const.
            obs, reward, terminated, truncated, info = env.step(action)
            action = action[1:]
            obs_onehot = _flatten_obs_onehot(obs, imep_space, mprr_space)       # flatten to one-hot shape needed by ort
            obs_flat = _flatten_obs_array(obs)                              # flatten to shape needed by memory buffer

            # log rollout into episode shm buffer
            try:
                # package rollout
                current_packet = np.concatenate((
                    action,
                    np.array([reward], dtype=np.float32),
                    obs_flat,
                    np.array([logp], dtype=np.float32),
                    logits_soi.astype(np.float32),
                    logits_d.astype(np.float32),
                )).astype(np.float32)

                # write it into the buffer
                write_fragment(
                    current_packet,
                    ep_arr,
                    f_buf,
                    episode_shm_properties,
                    logger=logger,
                    env=env,
                    reset_env_each_batch=True,      # would only apply when training using a gym env
                )
            except Exception as e:
                logger.error(f"Minion: could not write fragment due to error: {e}")

            # update policy network weights if available and not being written to
            if f_buf[1] == 1 and f_buf[0] == 0:
                logger.debug("Minion: Updating ort session weights...")
                f_buf[0] = 1  # set lock flag to locked
                ort_session, output_names = _get_ort_session(p_buf, logger=logger)  # get ort session with new weights
                f_buf[0] = 0  # set lock flag to unlocked
                f_buf[1] = 0  # reset weights-available flag to 0 (false, i.e. no new weights)
                logger.debug("Minion: Ort weights updated.")
                weight_updates += 1
                logger.debug(f"Minion: Update number -> {weight_updates}.")

            # set minion rollout flag to true to enable the algo.train() calls
            f_buf[2] = 1

            # logger.debug(f"Minion: Done with iteration {timesteps}")

            # if environment is the physical engine, wait for new state update and reward



    except KeyboardInterrupt:
        # close socket connection
        del ep_arr
        print("Program interrupted")

# if __name__ == '__main__':
#     import argparse
#
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--weights_shm_name", type=str, required=True, help="name of weights shm block")
#     parser.add_argument("--episodes_shm_name", type=str, required=True, help="name of episode shm block")
#     parser.add_argument("--flag_shm_name", type=str, required=True, help="name of flag shm block")
#     args = parser.parse_args()
#
#     main(args.weights_shm_name, args.flag_shm_name, args.episodes_shm_name)

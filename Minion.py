from multiprocessing import shared_memory
import numpy as np
import struct
import onnxruntime as ort
import time
import gzip

from ray.rllib.utils.numpy import softmax
import gymnasium as gym

from shared_memory_env_runner import (
    get_indices,
    set_indices
)


def _get_ort_session(buf: shared_memory.SharedMemory.buf):
    ort_compressed = buf[:]  # this is the actual content of p_buf (bytes)
    ort_bytes = gzip.decompress(ort_compressed)
    ort_session = ort.InferenceSession(
        ort_bytes,
        providers=["CoreMLExecutionProvider",
                   {"ModelFormat": "MLProgram", "MLComputeUnits": "ALL", "RequireStaticInputShapes": "1"}]
    )
    output_names = [o.name for o in ort_session.get_outputs()]
    return ort_session, output_names


def write_fragment(data: np.ndarray,
                   buf_arr: np.ndarray,
                   episode_shm_properties: dict,
                   *,
                   is_initial_state: bool = False,
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

    write_idx, read_idx = get_indices(buf_arr)
    slot_off = episode_shm_properties["HEADER_SIZE"] + write_idx * episode_shm_properties["SLOT_SIZE"]

    # only run this the very first time the episode buffer is started,
    # for other iterations the function already handles adding the initial state
    if is_initial_state:
        initial_state_off = slot_off + episode_shm_properties["HEADER_SLOT_SIZE"]
        buf_arr[initial_state_off: initial_state_off + len(data)] = data
        return buf_arr

    # get how many rollouts have been filled in the current episode
    filled = int(buf_arr[slot_off])

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
    if filled == episode_shm_properties["BATCH_SIZE"]:
        next_w = (write_idx + 1) % episode_shm_properties["NUM_SLOTS"]
        if next_w == read_idx:  # ring full → drop oldest
            read_idx = (read_idx + 1) % episode_shm_properties["NUM_SLOTS"]

        # get offset for next slot
        write_idx = next_w
        slot_off = episode_shm_properties["HEADER_SIZE"] + write_idx * episode_shm_properties["SLOT_SIZE"]

        # extract last state from current rollout (raw)
        state_off = len(data) - episode_shm_properties["STATE_ACTION_DIMS"]['state']
        state = data[state_off:]

        # add starting state to next slot
        initial_state_off = slot_off + episode_shm_properties["HEADER_SLOT_SIZE"]
        buf_arr[initial_state_off: initial_state_off + len(state)] = state

        # reset the fill counter to include initial state
        buf_arr[slot_off] = 0

    # Commit updated indices
    set_indices(buf_arr, write_idx, read_idx)

    return buf_arr


def main(policy_shm_name: str,
         flag_shm_name: str,
         ep_arr: np.ndarray,
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
    # connect to shared memory
    f_shm = shared_memory.SharedMemory(name=flag_shm_name, create=False)  # this one has to be first
    f_buf = f_shm.buf
    while f_buf[0] == 1:  # wait until other shared memory blocks have been created
        time.sleep(0.01)

    p_shm = shared_memory.SharedMemory(name=policy_shm_name, create=False)

    # get buffers
    p_buf = p_shm.buf

    # get initial network weights
    while f_buf[1] == 0:  # wait until weights-available flag is set to true
        time.sleep(0.01)
    ort_session, output_names = _get_ort_session(p_buf)
    f_buf[1] = 0  # change new-weights-available flag to false

    # initialize environment (gym.Env or socket to LabVIEW)
    # get env.reset() also, meaning system's initial state observation
    env = gym.make("CartPole-v1")
    obs, info = env.reset()

    # write initial state to episode buffer
    write_fragment(obs,
                   ep_arr,
                   episode_shm_properties,
                   is_initial_state=True)

    timesteps = 0

    try:
        while True:
            timesteps += 1

            # perform action inference using the ort model
            logits = ort_session.run(
                output_names,
                {"obs": np.array([obs], np.float32)},
            )[0][0]  # first [0] -> selects "output". second [0] -> selects 0th batch

            # stochastic sample of actions (for discrete action space only)
            action_probs = softmax(logits)
            action = int(np.random.choice(list(range(env.action_space.n)), p=action_probs))
            logp = float(np.log(action_probs[action]))

            # send action to environment (and in the case of gym.Env collect new observation and reward)
            obs, reward, terminated, truncated, info = env.step(action)

            # write data from current rollout to episode buffer
            current_packet = np.array([action, reward, obs], dtype=np.float32)
            write_fragment(current_packet, ep_arr, episode_shm_properties, )

            # update policy network weights if available and not being written to
            if f_buf[1] == 1 and f_buf[0] == 0:
                f_buf[0] = 1  # set lock flag to locked
                ort_session, output_names = _get_ort_session(p_buf)  # get ort session with new weights
                f_buf[0] = 0  # set lock flag to unlocked
                f_buf[1] = 0  # reset weights-available flag to 0 (false, i.e. no new weights)

            # if environment is the physical engine, wait for new state update and reward



    except KeyboardInterrupt:
        # close socket connection
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

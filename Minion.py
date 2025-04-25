from multiprocessing import shared_memory
import numpy as np

def _minion(weights_shm_name: str='weights',
            flag_shm_name: str='flag'):

    # connect to shared memory
    w_shm = shared_memory.SharedMemory(name=weights_shm_name)
    weights = np.ndarray((256, 128), dtype=np.float32, buffer=w_shm.buf)

    f_shm = shared_memory.SharedMemory(name=flag_shm_name)
    flag = np.ndarray((), dtype=np.uint8, buffer=f_shm.buf)




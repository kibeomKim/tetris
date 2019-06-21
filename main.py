import os
os.environ["OMP_NUM_THREADS"] = "1"
import torch
import torch.multiprocessing as mp
import torch.optim as optim

from run import run_loop
from test import test
from shared_optim import SharedRMSprop, SharedAdam
from model import A3C, A3C_LSTM

class Params():
    def __init__(self):
        self.n_process = 14
        self.episode = 100000000
        self.episode_test = 9999999
        self.lr = 0.001
        self.amsgrad = True
        self.weight_decay = 0
        self.gpu_ids_train = [0, 1, 2]
        self.gpu_ids_test = [3]
        self.num_steps = 100    # update per num_steps actions
        self.gamma = 0.99
        self.entropy_coef = 0.1
        self.value_loss_coef = 0.5
        self.tau = 1.0

if __name__ == "__main__":
    params = Params()
    mp.set_start_method('spawn')
    count = mp.Value('i', 0)    # update count
    lock = mp.Lock()

    # shared_model = A3C()
    shared_model = A3C_LSTM()
    shared_model = shared_model.share_memory()

    # shared_optimizer = SharedAdam(shared_model.parameters(), lr=params.lr, amsgrad=params.amsgrad,
    #                               weight_decay=params.weight_decay)
    shared_optimizer = SharedRMSprop(shared_model.parameters(), lr=params.lr)
    shared_optimizer.share_memory()

    # run_loop(0, params, shared_model, shared_optimizer, count, lock)    # for debugging
    # test(0, params, shared_model, count, lock)

    processes = []

    # have to add test module
    p = mp.Process(target=test, args=(0, params, shared_model, count, lock, ))
    p.start()
    processes.append(p)

    for rank in range(params.n_process):
        p = mp.Process(target=run_loop, args=(rank, params, shared_model, shared_optimizer, count, lock, ))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
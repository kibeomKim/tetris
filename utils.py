import torch
import copy
import numpy as np

import pdb

def ensure_shared_grads(model, shared_model, gpu=False):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None and not gpu:
            return
        elif not gpu:
            shared_param._grad = param.grad
        else:
            shared_param._grad = param.grad.cpu()

def pre_processing(curr_map,curr_block_pos):
    # ret = [[0] * 84 for _ in range(84)]

    block_pos = [[0] * 8 for _ in range(20)]
    copy_map = copy.deepcopy(curr_map)

    for n in curr_block_pos:
        block_pos[n[0]][n[1]] = 1
        # copy_map[n[0]][n[1]] = 1

    copy_map = np.array(copy_map, dtype='f')
    block_pos = np.array(block_pos, dtype='f')

    state = np.stack([copy_map, block_pos])
    # ny, nx = 4.20, 10.5
    # for n in curr_block_pos:
    #     copy_map[n[0]][n[1]] = 1
    # for n in range(20):
    #     for m in range(8):
    #         for i in range(int(n * ny), int(n * ny + ny)):
    #             for j in range(int(m * nx), int(m * nx + nx)):
    #                 ret[i][j] = copy_map[n][m]

    # return ret
    return state
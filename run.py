import torch

# from tetris import Env
from tetris_shadow import Env
from model import A3C, A3C_LSTM
from agent import run_agent

from utils import pre_processing
import random
from setproctitle import setproctitle as ptitle
import numpy as np

import pdb

def run_loop(rank, params, shared_model, shared_optimizer, count, lock):
    ptitle('Training Process: {}'.format(rank))
    gpu_id = params.gpu_ids_train[rank % len(params.gpu_ids_train)]

    env = Env(False, 1, down_period=2)

    # model = A3C()
    model = A3C_LSTM()
    with torch.cuda.device(gpu_id):
        model = model.cuda()
    agent = run_agent(model, gpu_id)

    episode = 0
    while episode <= params.episode:
        env.reset()
        agent.done = False
        num_steps = 0

        agent.synchronize(shared_model)
        nAction = 0

        nMove = 0

        while True:
            num_steps += 1
            # random_action = random.randrange(0, 5)
            '''
            if nAction < 9:
                obs = pre_processing(env.map, env._get_curr_block_pos())
                action, value, log_prob, entropy = agent.action_train(obs)
                rew, is_new_block = env.step(action)     # what is the 'is_new_block'?
                nAction += 1
                if nAction != 9:
                    rew = np.clip(rew, 0.0, 64.0)
                    agent.put_reward(rew, value, log_prob, entropy)
            else:
                rew, is_new_block = env.step(100000)  # falling
                rew = np.clip(rew, 0.0, 64.0)
                agent.put_reward(rew, value, log_prob, entropy)
                nAction = 0
            '''
            obs = pre_processing(env.shadow_map, env._get_curr_block_pos())   # env.map
            action, value, log_prob, entropy = agent.action_train(obs)
            if action == 5:
                action = 100000
            rew, shadow_rew, done, putting, height = env.step(action)  # what is the 'is_new_block'?
            rew = np.clip(rew, -1.0, 64.0)
            if rew == 0.0 and action != 3 and action != 4:
                nMove += 1
                if nMove < 6:
                    rew = 0.2
                if putting:
                    rew = - (height / 20.0)
                    nMove = 0
            agent.put_reward(rew, value, log_prob, entropy)

            # pdb.set_trace()
            if env.is_game_end():
                episode += 1
                agent.done = True

            # if num_steps % params.num_steps == 0:
            # if env.is_game_end() or rew >= 1.0:
            if env.is_game_end():
                next_obs = pre_processing(env.map, env._get_curr_block_pos())
                agent.training(next_obs, shared_model, shared_optimizer, params)
                with lock:  # synchronize vale of all process
                    count.value += 1

            if env.is_game_end():
                break



import torch

# from tetris import Env
from tetris_shadow import Env
from model import A3C, A3C_LSTM
from agent import run_agent
from utils import pre_processing

import random
from setproctitle import setproctitle as ptitle
import time
import pdb
import logging

def test(rank, params, shared_model, count, lock):
    logging.basicConfig(filename='./2blocks_rew.log', level=logging.INFO)
    ptitle('Test Process: {}'.format(rank))
    gpu_id = params.gpu_ids_test[rank % len(params.gpu_ids_test)]

    env = Env(True, 1, down_period=2)

    # model = A3C()
    model = A3C_LSTM()
    with torch.cuda.device(gpu_id):
        model = model.cuda()
    agent = run_agent(model, gpu_id)

    episode = 0
    while episode <= params.episode_test:
        env.reset()
        with lock:
            n_update = count.value
        agent.synchronize(shared_model)

        num_steps = 0
        accumulated_reward = 0
        nAction = 0
        line1 = 0
        line2 = 0
        line3 = 0
        line4 = 0
        nMove = 0
        rew_height = 0
        rew_move = 0

        while True:
            num_steps += 1

            obs = pre_processing(env.shadow_map, env._get_curr_block_pos())   # env.map
            action = agent.action_test(obs)
            if action == 5:
                action = 100000
            rew, shadow_reward, done, putting, height = env.step(action)     # what is the 'is_new_block'?
            if rew == 0.0 and action != 3 and action != 4:
                nMove += 1
                if nMove < 6:
                    rew_move += 0.2
                if putting:
                    rew_height += - (height / 20.0)
                    nMove = 0

            if rew  == 1.0:
                line1 += 1
            elif rew == 8.0:
                line2 += 1
            elif rew == 27.0:
                line3 += 1
            elif rew == 64:
                line4 += 1
            '''
            if nAction < 9:
                obs = pre_processing(env.map, env._get_curr_block_pos())
                action = agent.action_test(obs)
                rew, shadow_reward, is_new_block = env.step(action)     # what is the 'is_new_block'?
                nAction += 1

            else:
                rew, is_new_block = env.step(100000)  # falling
                nAction = 0
            '''

            accumulated_reward = rew + rew_move + rew_height

            if env.is_game_end():
                episode += 1
                print(" ".join([
                    "-------------episode stats-------------\n",
                    "nUpdate: {}\n".format(n_update),
                    "line1: {}\n".format(line1),
                    "line2: {}\n".format(line2),
                    "line3: {}\n".format(line3),
                    "line4: {}\n".format(line4),
                    "all_lines: {}\n".format(str(line1 + line2 + line3 + line4)),
                    "score: {}\n".format(env.score),
                    "rew_move: {}\n".format(rew_move),
                    "rew_height: {}\n".format(rew_height),
                    "steps: {}\n".format(num_steps)
                ]))
                logging.info(" ".join([
                    "-------------episode stats-------------\n",
                    "nUpdate: {}\n".format(n_update),
                    "line1: {}\n".format(line1),
                    "line2: {}\n".format(line2),
                    "line3: {}\n".format(line3),
                    "line4: {}\n".format(line4),
                    "all_lines: {}\n".format(str(line1 + line2 + line3 + line4)),
                    "score: {}\n".format(env.score),
                    "rew_move: {}\n".format(rew_move),
                    "rew_height: {}\n".format(rew_height),
                    "steps: {}\n".format(num_steps)
                ]))
                break

            if env.score > 1000:
                episode += 1
                print(" ".join([
                    "-------------episode stats-------------\n",
                    "nUpdate: {}\n".format(n_update),
                    "line1: {}\n".format(line1),
                    "line2: {}\n".format(line2),
                    "line3: {}\n".format(line3),
                    "line4: {}\n".format(line4),
                    "all_lines: {}\n".format(str(line1+line2+line3+line4)),
                    "score: {}\n".format(env.score),
                    "rew_move: {}\n".format(rew_move),
                    "rew_height: {}\n".format(rew_height),
                    "steps: {}\n".format(num_steps)
                ]))
                with torch.cuda.device(gpu_id):
                    torch.save(agent.model.state_dict(), './weight/model' + str(n_update) + '.ckpt')
                logging.info(" ".join([
                    "-------------episode stats-------------\n",
                    "nUpdate: {}\n".format(n_update),
                    "line1: {}\n".format(line1),
                    "line2: {}\n".format(line2),
                    "line3: {}\n".format(line3),
                    "line4: {}\n".format(line4),
                    "all_lines: {}\n".format(str(line1+line2+line3+line4)),
                    "score: {}\n".format(env.score),
                    "rew_move: {}\n".format(rew_move),
                    "rew_height: {}\n".format(rew_height),
                    "steps: {}\n".format(num_steps)
                ]))
                break
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_

import numpy as np
from utils import ensure_shared_grads

import pdb

# np.set_printoptions(threshold=np.nan)

def preprocessing(obs, obs_old, gpu_id):
    """
    :param obs: [channel=2, width=20, height=8]
    :return: [batch size, channel, width, height]
    """
    # pdb.set_trace()
    state = np.concatenate((obs, obs_old))
    state = torch.from_numpy(state)
    state = state.unsqueeze(0)

    with torch.cuda.device(gpu_id):
        state = Variable(torch.FloatTensor(state)).cuda()

    return state


class run_agent(object):
    def __init__(self, model, gpu_id):
        self.model = model
        self.gpu_id = gpu_id
        self.done = False
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        self.hx, self.cx = None, None
        self.obs_old = None

    def action_train(self, obs):
        """
        :param obs: [width=20, height=8]
        :return:
        """
        if self.obs_old is None:
            self.obs_old = obs
        state = preprocessing(obs, self.obs_old, self.gpu_id)

        # value, logit = self.model(state)
        value, logit, self.hx, self.cx = self.model(state, self.hx, self.cx)

        prob = F.softmax(logit, dim=1)
        log_prob = F.log_softmax(logit, dim=1)
        entropy = -(log_prob * prob).sum(1)

        action = prob.multinomial(1)
        # action = prob.max(1)[1].unsqueeze(0)
        log_prob = log_prob.gather(1, Variable(action.data))

        # self.values.append(value)
        # self.log_probs.append(log_prob)
        # self.entropies.append(entropy)
        self.obs_old = obs

        return action.cpu().numpy().squeeze(0)[0], value, log_prob, entropy
        # return action.data.cpu().numpy()[0][0], value, log_prob, entropy

    def action_test(self, obs):
        """
        :param obs: [width=20, height=8]
        :return:
        """
        if self.obs_old is None:
            self.obs_old = obs
        state = preprocessing(obs, self.obs_old, self.gpu_id)

        # value, logit = self.model(state)
        value, logit, self.hx, self.cx = self.model(state, self.hx, self.cx)

        self.obs_old = obs
        prob = F.softmax(logit, dim=1)

        action = prob.max(1)[1].data.cpu().numpy()[0]

        return action

    def clear_all(self):
        self.values.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.entropies.clear()

    def put_reward(self, reward, value, log_prob, entropy):
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.entropies.append(entropy)

    def synchronize(self, shared_model):
        with torch.cuda.device(self.gpu_id):
            self.model.load_state_dict(shared_model.state_dict())
            self.cx = Variable(torch.zeros(1, 256).cuda())
            self.hx = Variable(torch.zeros(1, 256).cuda())
        self.obs_old = None


    def training(self, next_obs, shared_model, shared_optimizer, params):
        #pdb.set_trace()
        # self.model.train()
        self.cx = Variable(self.cx.data)
        self.hx = Variable(self.hx.data)

        R = torch.zeros(1, 1)
        if not self.done:
            state = preprocessing(next_obs, self.obs_old, self.gpu_id)
            value, _, _, _ = self.model(state, self.hx, self.cx)
            R = value.data

        with torch.cuda.device(self.gpu_id):
            R = R.cuda()
        R = Variable(R)
        self.values.append(R)

        policy_loss = 0
        value_loss = 0
        gae = torch.zeros(1, 1)

        with torch.cuda.device(self.gpu_id):
            gae = gae.cuda()

        for i in reversed(range(len(self.rewards))):
            R = params.gamma * R + self.rewards[i]
            advantage = R - self.values[i]
            value_loss = value_loss + advantage.pow(2)  # 0.5 *

            # Generalized Advantage Estimation
            delta_t = params.gamma * self.values[i + 1].data - self.values[i].data + self.rewards[i]

            gae = gae * params.gamma * params.tau + delta_t

            policy_loss = policy_loss - self.log_probs[i] * Variable(gae) - params.entropy_coef * self.entropies[i]

        shared_optimizer.zero_grad()
        loss = policy_loss + params.value_loss_coef * value_loss
        loss.backward()

        clip_grad_norm_(self.model.parameters(), 50.0)
        ensure_shared_grads(self.model, shared_model, gpu=self.gpu_id>=0)
        shared_optimizer.step()

        # self.synchronize(shared_model)
        with torch.cuda.device(self.gpu_id):
            self.model.load_state_dict(shared_model.state_dict())
        self.clear_all()


import numpy as np
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimum

from tennis.models import Actor, Critic
from tennis.noise import OUNoise


TAU = 1e-2
LR_ACTOR = 2e-4
LR_CRITIC = 2e-4
POLICY_NOISE = 0.2
NOISE_CLIP =0.5
WEIGHT_DECAY = 0.0 #L2 weight decay

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class Agent(object):
    """
    Base Agent
    """
    def __init__(self,state_dim, action_dim,seed):
        """
        Initialize Agent object
        """

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.seed = random.seed(seed)

        #Initialize Actor Network (Local / Target networks)
        self.actor_local = Actor(state_dim,action_dim,seed).to(device)
        self.actor_target = Actor(state_dim,action_dim,seed).to(device)
        self.actor_target.load_state_dict(self.actor_local.state_dict())
        self.actor_optimizer =optimum.Adam(self.actor_local.parameters(),lr=LR_ACTOR)

        # Twin Critic Network(s) (Local / Target networks)
        self.critic_local = Critic(state_dim,action_dim,seed).to(device)
        self.critic_target = Critic(state_dim,action_dim,seed).to(device)
        self.critic_target.load_state_dict(self.critic_local.state_dict())
        self.critic_optimizer = optimum.Adam(self.critic_local.parameters(),lr=LR_CRITIC,weight_decay=WEIGHT_DECAY)

        # Initialize Ornstein-Uhlenbeck Noise process
        self.noise = OUNoise(action_dim,seed,NOISE_CLIP)

    def reset(self):
        """
        To Reset noise process to mean
        """
        self.noise.reset()


    def act(self,state,add_noise=True):
        """
        Returns a deterministic action given current state.
        :state: current state
        :add_noise: add bias to agent. Default = True
        """
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()

        if add_noise:
            action += self.noise.sample()
        return np.clip(action,-1,1)

    def learn(self,experiences, gamma):
        """
        to learn from the set of experiences
        :experiences: set of experiences, trajectory, tau. tuple of (s,a,r,s',done)
        :gamma: immediate reward hyperparameter
        """
        states, actions, rewards, next_states, dones = experiences

        # --------- Update Critic network -------#
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states,actions_next)
        Q_targets = rewards + (gamma * Q_targets_next * (1- dones))

        #--------- Compute critic loss using MSE--#
        Q_expected = self.critic_local(states,actions)
        critic_loss = F.mse_loss(Q_expected,Q_targets)

        #----------Minimize the loss -----------#
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm(self.critic_local.parameters(),1)
        self.critic_optimizer.step()

        #-------- Update Actor network---------#
        # get mu(s)
        actions_pred = self.actor_local(states)
        #get V(s,a)
        actor_loss = -self.critic_local(states,actions_pred).mean()
        #-------- Minize the loss -----------#
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        #------- Update Target Networks ----#
        self.soft_update(self.critic_local,self.critic_target,TAU)
        self.soft_update(self.actor_local,self.actor_target,TAU)


    def soft_update(self, local_model, target_model, tau):
        """
        Soft update model parameters
        θ_target = τ*θ_local + (1 - τ)*θ_target
        :param local_model:
        :param target_model:
        :param tau:
        :return:
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)




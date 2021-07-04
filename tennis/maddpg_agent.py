from tennis.ddpg_agent import Agent
from tennis.storage import ReplayBuffer
import os

import torch

# Define Constants
BATCH_SIZE = 256
BUFFER_SIZE = int(1e5)
GAMMA = 0.9


class MADDPG(object):

    def __init__(self, num_agents=2, state_size=24, action_size=2, random_seed=40):
        """
        Initializes multiple agents using DDPG algorithm
        :num_agents: number of agents provided by the unity brain
        :state_size:
        :action_size:
        random_seed:
        """
        self.num_agents = num_agents
        # ----------- Create n agents, where n = num_agents ---------#
        self.agents = [Agent(state_size, action_size, random_seed) for _ in range(self.num_agents)]

        # ---------- Create shared experience replay buffer --------#
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed=random_seed)

    def act(self, states, add_noise=True):
        """
        Perform action for mulitple agents. Uses single agent.act()
        """
        actions = []
        for state, agent in zip(states, self.agents):
            action = agent.act(state, add_noise)
            actions.append(action)
        return actions

    def reset(self):
        """
        Reset the noise level of mulitple agents
        """
        for agent in self.agents:
            agent.reset()

    def step(self, states, actions, rewards, next_states, dones):
        """
        Saves an experience in replay buffer to learn from random sample
        :states:
        :actions:
        :rewards:
        :next_states:
        :dones:
        """
        for i in range(self.num_agents):
            self.memory.add(states[i], actions[i], rewards[i], next_states[i], dones[i])

        # ------ check if enough samples in buffer ------#
        if len(self.memory) > BATCH_SIZE:
            for _ in range(self.num_agents):
                expereince = self.memory.sample()
                self.learn(expereince)

    def learn(self, experiences, gamma=GAMMA):
        for agent in self.agents:
            agent.learn(experiences, gamma)

    def saveCheckPoints(self, isDone):
        """Save the checkpoint weights of MARL params every 100 or so episodes"""
        if (isDone == False):
            for i, agent in enumerate(self.agents):
                torch.save(agent.actor_local.state_dict(), f"../pytorch_models/checkpoint/actor_agent_{i}.pth")
                torch.save(agent.critic_local.state_dict(), f"../pytorch_models/checkpoint/critic_agent_{i}.pth")
        else:
            for i, agent in enumerate(self.agents):
                torch.save(agent.actor_local.state_dict(), f"../pytorch_models/final/actor_agent_{i}.pth")
                torch.save(agent.critic_local.state_dict(), f"../pytorch_models/final/critic_agent_{i}.pth")

    def loadCheckPoints(self, isFinal=False):
        """Loads the checkpoint weight of MARL params"""
        if (isFinal):
            for i, agent in enumerate(self.agents):
                agent.actor_local.load_state_dict(torch.load(f"./pytorch_models/final/actor_agent_{i}.pth"))
                agent.critic_local.load_state_dict(torch.load(f"./pytorch_models/final/critic_agent_{i}.pth"))
        else:
            for i, agent in enumerate(self.agents):
                agent.actor_local.load_state_dict(torch.load(f"../pytorch_models/checkpoint/actor_agent_{i}.pth"))
                agent.critic_local.load_state_dict(torch.load(f"../pytorch_models/checkpoint/critic_agent_{i}.pth"))


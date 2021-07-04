import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """
        Estimates the policy deterministically using tanh activation for continuous action space
    """

    def __init__(self,state_dim,action_dim,seed,fc1_units=128,fc2_units=128):
        """

        :param state_dim:
        :param action_dim:
        :param seed:
        :param fc1_units:
        :param fc2_units:
        """
        super(Actor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_dim,fc1_units)
        self.bn1 = nn.BatchNorm1d(fc1_units)
        # Second Layer
        self.fc2 = nn.Linear(fc1_units,fc2_units)
        self.bn2 = nn.BatchNorm1d(fc2_units)

        self.fc3 = nn.Linear(fc2_units,action_dim)

        self.reset_parameters()


    def reset_parameters(self):
        """
        :return:
        """
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self,state):
        """
         Performs a single forward pass to map (state,action) to policy, pi.
        :param state:
        :return:
        """
        if state.dim() ==1:
            state = torch.unsqueeze(state,0)
        x = state
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)

        # Layer #2
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)

        # Output
        x = self.fc3(x)
        mu = torch.tanh(x)
        return mu

class Critic(nn.Module):
    """
    Value approximator V(pi) as Q(s, a|θ)
    """
    def __init__(self,state_dim,action_dim,seed,fc1_units=128,fc2_units=128):
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        # Layer 1
        self.fc1 = nn.Linear(state_dim, fc1_units)
        self.bn1 = nn.BatchNorm1d(fc1_units)
        # Layer 2
        self.fc2 = nn.Linear(fc1_units + action_dim, fc2_units)
        # Output layer
        self.fc3 = nn.Linear(fc2_units, 1)  # Q-value

        # Initialize Weights
        self.reset_parameters()

    def reset_parameters(self):
        """
        Resets the parameters by setting a noise from distribution following from its respective hidden unit size.
        Format for (-fx,fx) followed from the original paper.
        """
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """
        Performs a single forward pass to map (state,action) to Q-value
        @Param:
        1. state: current observations, shape: (env.observation_space.shape[0],)
        2. action: immediate action to evaluate against, shape: (env.action_space.shape[0],)
        @Return:
        - q-value
        """
        if state.dim() == 1:
            state = torch.unsqueeze(state, 0)
        # Layer #1
        x = self.fc1(state)
        x = F.relu(x)
        x = self.bn1(x)
        # Layer #2
        x = torch.cat((x, action),
                      dim=1)  # Concatenate state with action. Note that the specific way of passing x_state into layer #2.
        x = self.fc2(x)
        x = F.relu(x)
        # Output
        value = self.fc3(x)
        return value








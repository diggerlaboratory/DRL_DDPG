import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS_FINAL_INIT = 3e-3
BIAS_FINAL_INIT = 3e-4


def fan_in_uniform_init(tensor, fan_in=None):
    """Utility function for initializing actor and critic"""
    if fan_in is None:
        fan_in = tensor.size(-1)

    w = 1. / np.sqrt(fan_in)
    nn.init.uniform_(tensor, -w, w)


class Actor(nn.Module):
    def __init__(self, hidden_size, num_inputs, action_space):
        super(Actor, self).__init__()
        self.action_space = action_space
        num_outputs = action_space.shape[0]

        # Layer 1
        self.linear1 = nn.Linear(num_inputs, hidden_size[0])
        self.ln1 = nn.LayerNorm(hidden_size[0])

        self.linear2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.ln2 = nn.LayerNorm(hidden_size[1])

        self.linear3 = nn.Linear(hidden_size[1], hidden_size[2])
        self.ln3 = nn.LayerNorm(hidden_size[2])

        self.linear4 = nn.Linear(hidden_size[2], hidden_size[3])
        self.ln4 = nn.LayerNorm(hidden_size[3])

        self.linear5 = nn.Linear(hidden_size[3], hidden_size[4])
        self.ln5 = nn.LayerNorm(hidden_size[4])
        
        # Output Layer
        self.out = nn.Linear(hidden_size[4], num_outputs)

        # # Weight Init
        # fan_in_uniform_init(self.linear1.weight)
        # fan_in_uniform_init(self.linear1.bias)

        # fan_in_uniform_init(self.linear2.weight)
        # fan_in_uniform_init(self.linear2.bias)

        # nn.init.uniform_(self.out.weight, -WEIGHTS_FINAL_INIT, WEIGHTS_FINAL_INIT)
        # nn.init.uniform_(self.out.bias, -BIAS_FINAL_INIT, BIAS_FINAL_INIT)

    def forward(self, inputs):
        x = inputs
        # Layer 1
        x = self.linear1(x)
        x = self.ln1(x)
        x = F.relu(x)

        # Layer 2
        x = self.linear2(x)
        x = self.ln2(x)
        x = F.relu(x)

        # Layer 2
        x = self.linear3(x)
        x = self.ln3(x)
        x = F.relu(x)
        # Layer 2
        x = self.linear4(x)
        x = self.ln4(x)
        x = F.relu(x)
        # Layer 2
        x = self.linear5(x)
        x = self.ln5(x)
        x = F.relu(x)

        # Output
        out = torch.tanh(self.out(x))
        return out


class Critic(nn.Module):
    def __init__(self, hidden_size, num_inputs, action_space):
        super(Critic, self).__init__()
        self.action_space = action_space
        num_outputs = action_space.shape[0]

        # Layer 1
        self.linear1 = nn.Linear(num_inputs, hidden_size[0])
        self.ln1 = nn.LayerNorm(hidden_size[0])

        # Layer 1
        self.linear2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.ln2 = nn.LayerNorm(hidden_size[1])

        # Layer 2
        self.linear3 = nn.Linear(hidden_size[1], hidden_size[2])
        self.ln3 = nn.LayerNorm(hidden_size[2])
        # Layer 3
        self.linear4 = nn.Linear(hidden_size[2] + num_outputs, hidden_size[3])
        self.ln4 = nn.LayerNorm(hidden_size[3])

        # Layer 4
        self.linear5 = nn.Linear(hidden_size[3], hidden_size[4])
        self.ln5 = nn.LayerNorm(hidden_size[4])



        # Output layer (single value)
        self.V = nn.Linear(hidden_size[4], 1)

        # Weight Init
        # fan_in_uniform_init(self.linear1.weight)
        # fan_in_uniform_init(self.linear1.bias)

        # fan_in_uniform_init(self.linear2.weight)
        # fan_in_uniform_init(self.linear2.bias)

        # nn.init.uniform_(self.V.weight, -WEIGHTS_FINAL_INIT, WEIGHTS_FINAL_INIT)
        # nn.init.uniform_(self.V.bias, -BIAS_FINAL_INIT, BIAS_FINAL_INIT)

    def forward(self, inputs, actions):
        x = inputs
        # Layer 1
        x = self.linear1(x)
        x = self.ln1(x)
        x = F.relu(x)

        # Layer 2
        x = self.linear2(x)
        x = self.ln2(x)
        x = F.relu(x)

        # Layer 3
        x = self.linear3(x)
        x = self.ln3(x)
        x = F.relu(x)
        
        # Layer 4
        x = torch.cat((x, actions), 1)  # Insert the actions
        x = self.linear4(x)
        x = self.ln4(x)
        x = F.relu(x)
        
        # Layer 5
        x = self.linear5(x)
        x = self.ln5(x)
        x = F.relu(x)
        # Output
        V = self.V(x)
        return V

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Sieć Aktora dla wyznaczania Polityki."""

    def __init__(self, state_size, action_size, seed, fc1_units=400, fc2_units=300):
        """
        Argumenty funkcji
        ======
            state_size (int): Wymiary Stanu
            action_size (int): Wymiary każdej Akcji
            seed (int): Losowe ziarno dla powtarzalności rezultatów
            fc1_units (int): Liczba neuronów w pierwszej warstwie ukrytej
            fc2_units (int): Liczba neuronów w drugiej warstwie ukrytej
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Funkcja budująca sieć"""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return F.tanh(self.fc3(x))


class Critic(nn.Module):
    """Sieć Krytyka dla wyznaczania wartości Akcji."""

    def __init__(self, state_size, action_size, seed, fcs1_units=400, fc2_units=300):
        """
        Argumenty funkcji
        ======
            state_size (int): Wymiary Stanu
            action_size (int): Wymiary każdej Akcji
            seed (int): Losowe ziarno dla powtarzalności rezultatów
            fcs1_units (int): Liczba neuronów w pierwszej warstwie ukrytej
            fc2_units (int): Liczba neuronów w drugiej warstwie ukrytej
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear(state_size, fcs1_units)
        self.fc2 = nn.Linear(fcs1_units+action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Funkcja budująca sieć"""
        xs = F.relu(self.fcs1(state))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)



########################################################################################################################
# References:
# [1] Udacity, Deep Reinforcement Learning, Github, 2020, online: https://github.com/udacity/deep-reinforcement-learning
########################################################################################################################
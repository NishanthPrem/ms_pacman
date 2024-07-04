#%% Importing Libraries

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from torch.utils.data import DataLoader, TensorDataset

#%% Defining the Network class

class Network(nn.Module):
    def __init__(self, action_size, seed=42):
        super(Network, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.conv1 = nn.Conv2d(3,32,kernel_size=8,stride=4)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32,64,kernel_size=4,stride=2)
        self.bn2 = nn.BatchNorm2d(64)


        self.conv3 = nn.Conv2d(64,64,kernel_size=3,stride=1)
        self.bn3 = nn.BatchNorm2d(64)


        self.conv4 = nn.Conv2d(64,128,kernel_size=3,stride=1)
        self.bn4 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(10*10*128, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, action_size)

    def forward(self, state):
        out = F.relu(self.bn1(self.conv1(state)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.relu(self.bn3(self.conv3(out)))
        out = F.relu(self.bn4(self.conv4(out)))

        out = out.view(out.size(0),-1)

        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))

        return self.fc3(out)

#%% Setting up the environment

import gymnasium as gym
env = gym.make('MsPacmanDeterministic-v0', full_action_space=False)
state_shape = env.observation_space.shape
state_size = env.observation_space.shape[0]
number_action = env.action_space.n
print(state_shape)

#%% Initializing the hyperparameters

learning_rate = 5e-4
mini_batch_size = 64
discount_factor = 0.99

#%% Preprocessing the frames

from PIL import Image
from torchvision import transforms

def preprocess_frame(frame):
    frame = Image.fromarray(frame)
    preprocess = transforms.Compose([transforms.Resize((128,128)), transforms.ToTensor()])

    return preprocess(frame).unsqueeze(0)

#%% Setting up the agent class

class Agent():
    def __init__(self, action_size):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.action_size = action_size
        self.local_qnetwork = Network(action_size).to(self.device)
        self.target_qnetwork = Network(action_size).to(self.device)
        self.optimizer = optim.Adam(self.local_qnetwork, lr=learning_rate)
        self.memory = deque(maxlen=10000)

    def step(self, state, action, reward, next_state, done):
        state = preprocess_frame(state)
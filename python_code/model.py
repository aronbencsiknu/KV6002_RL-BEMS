import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, obs_count, action_count):
        super(DQN, self).__init__()

        # layers
        self.fc1 = nn.Linear(obs_count, 20)
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, 20)
        self.fc4 = nn.Linear(20, action_count)
        self.activation = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):

        # input layer
        x = self.fc1(x)
        x = self.activation(x)

        # 1st hidden layer
        x = self.fc2(x)
        #x = self.dropout(x)
        x = self.activation(x)

        # 2nd hidden layer
        x = self.fc3(x)
        #x = self.dropout(x)
        x = self.activation(x)

        # output layer
        x = self.fc4(x)

        return x

import torch


class Options(object):
    def __init__(self):
        self.episode_len = 5000
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.batch_size = 16
        self.num_episodes = 20
        self.gamma = 0.9
        self.beta = 0.3
        self.learning_rate = 0.001

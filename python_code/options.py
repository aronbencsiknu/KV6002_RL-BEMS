import torch


class Options(object):
    def __init__(self):
        self.episode_len = 5000
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.batch_size = 16
        self.num_episodes = 40
        self.gamma = 0.9  # foresight in bellman optimality equation
        self.beta = 0.3  # This is the epsilon decay rate
        self.learning_rate = 0.001
        self.num_epochs = 1
        self.demo_len = 5000
        self.pre_train = False
        self.demo_sleep = 1 # time.sleep in demo (in seconds)
        
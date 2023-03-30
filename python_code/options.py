import torch


class Options(object):
    def __init__(self):

        # episode length for pre-training
        self.episode_len = 5000

        # set device to GPU or CPU
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        
        # batch size
        self.batch_size = 16

        # define loss function
        self.loss_fn = torch.nn.SmoothL1Loss()

        # number of episodes for pre-training
        self.num_episodes = 100

        # foresight in bellman optimality equation
        self.gamma = 0.9  

        # This is the epsilon decay rate during pre-training
        self.beta = 0.1

        # learning rate for the optimizer
        self.learning_rate = 0.001

        # number of epochs in each episode iteration
        self.num_epochs = 1

        # length of the local demo
        self.local_demo_len = 5000

        # length of the GUI demo is set to maxInt
        self.gui_demo_len = 1000000

        # defines how much the simulation is slowed down in the GUI demo
        self.demo_sleep = 1

        # path to saved models from the root directory of the project
        self.path_to_model_from_root = "trained_models"

        # name to save a pre-trained model
        self.model_name_save = "test01"

        # name of the pre-trained model to load for local or GUI demo
        self.model_name_load = "example_er1p1"

        #  set wether to log data to Weights&Biases
        self.wandb = False
        self.wandb_key = "edfb94e4b9dca47c397a343d2829e9af262d9e32"
        self.wandb_logging_freq = 100
        
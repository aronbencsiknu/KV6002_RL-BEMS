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
        self.loss_fn = torch.nn.SmoothL1Loss()  # Huber loss

        # number of episodes for pre-training
        self.num_episodes = 60

        # foresight in bellman optimality equation
        self.gamma = 0.9  

        # This is the epsilon decay rate during pre-training
        self.beta = 0.4

        # learning rate for the optimizer
        self.learning_rate = 0.001
        self.demo_learning_rate = 0.0001

        # number of epochs in each episode iteration
        self.num_epochs = 1

        # length of the local demo
        self.local_demo_len = 10000

        # length of the GUI demo
        self.gui_demo_len = 10000000

        # change the desired temps in the reward function during training 
        # (intended to increase model flexibility to changing desired temps using demo)
        self.ch_rew = False

        # defines how much the simulation is slowed down in the GUI demo
        self.demo_sleep = 0.1

        # path to saved models from the root directory of the project
        self.path_to_model_from_root = "trained_models"

        # name to save a pre-trained model
        self.model_name_save = "test"

        # name of the pre-trained model to load for local or GUI demo
        self.model_name_load = "e60_er13_eps04_small_net_lrsch01"

        #  set wether to log data to Weights&Biases
        self.wandb = False
        self.wandb_key = "PASTE_KEY_HERE"
        self.wandb_logging_freq = 200
        
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from collections import deque
import random
from pylab import rcParams
#from tqdm import tqdm
from progress.bar import Bar

#LAKEvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
with open('room1.json', 'r') as f:
  data = json.load(f)
#LAKE^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

from environment import Environment  # import environment simulation
from options import Options  # import options
from reward import Reward  # import reward mechanism for agent
#from plot import Plot
opt = Options()

#LAKEvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
"""opt.num_episodes = 20
opt.episode_len = 5000"""
#LAKE^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


# we can declare max/min temps here, but we can also change them later via a setter method in Reward
reward = Reward()

#LAKEvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
"""reward.min_temp = int(data['minTemp'])
reward.max_temp = int(data['maxTemp'])
reward.max_allowed_temp_change = int(data['rateOfChange'])
reward.crit_min_temp = int(data['critMinTemp'])
reward.crit_max_temp = int(data['critMaxTemp'])
reward.crit_time = int(data['maxTime'])
"""#LAKE^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^



def experience_replay(model, batch_size, memory, obs_count, epoch_count):
    """
    Creates the IIDD supervised data from the experience memory, to train the DQN

    :param model: DQN network
    :param batch_size: size of random samples taken from memory
    :param memory: experience memory
    :param obs_count: observation size of the environment
    :param epoch_count: epoch count for training the DQN
    :return:
    """
    with torch.no_grad():
        batch = random.sample(memory, batch_size) # get random batch from memory with size var:batch_size
        batch_vector = np.array(batch, dtype=object)

        # create np arrays with correct shapes
        obs_t = np.zeros(shape=(batch_size, obs_count))
        obs_t_next = np.zeros(shape=(batch_size, obs_count))

        for i in range(len(batch_vector)):
            obs_t[i] = batch_vector[i, 0]
            obs_t_next[i] = batch_vector[i, 3]

        prediction_at_t = model(torch.tensor(obs_t).float().to(opt.device)).to("cpu")  # Use the model to predict an action using observations at t
        prediction_at_t_next = model(torch.tensor(obs_t_next).float().to(opt.device)).to("cpu")  # Use the model to predict an action using observations at t+1

        X = []
        y = []
        i = 0

        for obs_t, action, reward_value, _ in batch_vector:

            X.append(obs_t)

            # bellman optimality equation
            target = reward_value + opt.gamma * np.max(prediction_at_t_next[i].numpy())

            # update action  value
            prediction_at_t[i, action] = target

            y.append(
                prediction_at_t[i].numpy())

            i += 1  # increment i
        X = np.array(X).reshape(batch_size, obs_count)  # reshape X
        y = np.array(y) # create a numpy array from y

        loss = []

    for epoch in range(epoch_count):
        # Forward pass
        y_pred = model(torch.tensor(X).float().to(opt.device))
        loss_val = model.loss_fn(y_pred, torch.tensor(y).to(opt.device))

        loss.append(loss_val.item())
        # Backward pass and optimization step
        model.optimizer.zero_grad()
        loss_val.backward()
        model.optimizer.step()

    return loss  # return the loss


def list_window_averaging(win_len, list_to_avg):
    return_list = []
    current_window = 0
    current_window_sum = 0

    for i in list_to_avg:
        current_window += 1
        current_window_sum += i

        if current_window >= win_len:
            current_window = 0
            return_list.append([current_window_sum / win_len] * win_len)
            current_window_sum = 0

    return np.asarray(return_list).flatten()


def plot(world):
    rcParams['figure.figsize'] = 20, 5

    plt.plot(world.H_temp, label='Outside temperature', linewidth='10', color="blue")

    plt.plot([light * max(world.H_greenhouse_temp) for light in world.H_sunlight],
             label='Sunlight', linewidth='2', color="orange")

    plt.plot(world.H_greenhouse_temp, label='Greenhouse temperature', linewidth='3', color="green")

    #lake below

    heating_plot = list_window_averaging(win_len=50, list_to_avg=world.H_greenhouse_heating)
    ventilation_plot = list_window_averaging(win_len=50, list_to_avg=world.H_greenhouse_ventilation)

    plt.plot([energy * max(world.H_greenhouse_temp) for energy in heating_plot],
             label='Heating', linewidth='2', color="red")

    plt.plot([cooler*max(world.H_greenhouse_temp) for cooler in ventilation_plot], label = "Ventilation", linewidth ='2', color="black")
    # plt.figure(figsize=(10, 5))
    custom_ticks, custom_tick_names = world.get_custom_xcticks(world.H_temp)
    plt.xticks(custom_ticks, custom_tick_names)
    # custom_xticks = get_custom_xticks(len(world.H_greenhouse_temp))
    plt.legend()
    plt.show()


environment = Environment(
    0.1,  # cloudiness
    0.5)  # energy_consumption
observation = environment.get_state()
obs_count = len(observation)
action_count = 3


class DQN(nn.Module):
    def __init__(self, obs_count, action_count):
        super(DQN, self).__init__()

        # layers
        self.fc1 = nn.Linear(obs_count, 20)
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, 20)
        self.fc4 = nn.Linear(20, action_count)
        self.activation = nn.LeakyReLU()

        # training options
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=opt.learning_rate)  # AdamW optimizer

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.fc3(x)
        x = self.activation(x)
        x = self.fc4(x)
        return x


model = DQN(obs_count, action_count).to(opt.device)

rewards = []
loss = []
epsilons = []
beta = opt.beta  # This is the epsilon decay rate
batch_size = opt.batch_size
memory = deque([], maxlen=2500)

for episode in range(opt.num_episodes):
    environment = Environment(
        0.1,  # cloudiness
        0.5)  # energy_consumption
    obs_t = environment.get_state()
    total_reward = 0

    epsilon = 1 / (1 + opt.beta * (episode / action_count))
    epsilons.append(epsilon)

    bar_title = "Episode " + str(episode+1) + " of " + str(opt.num_episodes)
    bar = Bar(bar_title, max=opt.episode_len)

    for ep_index in range(opt.episode_len):
        with torch.no_grad():
            rand_num = np.random.random()

            # explore
            if rand_num <= epsilon:
                action_index = random.randint(0, action_count-1)

            # exploit
            else:
                action = model(torch.tensor(obs_t).float().to(opt.device)).to("cpu")
                action_index = np.argmax(action)

            # set actions
            if action_index == 0:
                heating = True
                ventilation = False
            elif action_index == 1:
                heating = False
                ventilation = True
            else:
                heating = False
                ventilation = False

            # advance simulation with actions
            environment.run(heating=heating, cooling=ventilation, steps=1, output_format='none')

            # get environment state
            obs_t_next = environment.get_state()

            # calculate reward (will be a call to another file)
            reward_value = reward.calculate_reward(environment.greenhouse.temp, environment.H_temp, heating) # input current variables here
            total_reward += reward_value

            # append to experience memory
            memory.append((obs_t, action_index, reward_value, obs_t_next))
            obs_t = obs_t_next

        # train DQN and calculate loss
        if len(memory) > batch_size:
            loss = experience_replay(model, batch_size, memory, obs_count, opt.num_epochs)

        bar.next()
    bar.finish()

    rewards.append(total_reward)
    avg_reward = total_reward/opt.episode_len
    print("\t - avg reward: %.4f" % avg_reward, "\n"
          "\t - avg loss: %.4f" % np.mean(np.asarray(loss)), "\n"
          "\t - epsilon: %.4f" % epsilon,"\n")

plot(environment)

#LAKEvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

with open('data.json', 'w') as f:
    json.dump(data, f)

#LAKE^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

#torch.save(model, 'test01')
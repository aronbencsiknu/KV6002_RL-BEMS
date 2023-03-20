import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from collections import deque
import random
from pylab import rcParams
from tqdm import tqdm
from environment import Environment  #import environment simulation
from options import Options
opt = Options()


def experience_replay(model, batch_size, memory, obs_count, action_count, epoch_count, loss):
    """
    Creates the IIDD supervised data from the experience memory, to train the DQN

    :param model: DQN network
    :param batch_size: size of random samples taken from memory
    :param memory: experience memory
    :param obs_count: observation size of the environment
    :param action_count: possible actions in the environment
    :param epoch_count: epoch count for training the DQN
    :param loss: loss of the DQN
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

        prediction_at_t = model(torch.tensor(obs_t).float().to("cuda")).to("cpu")  # Use the model to predict an action using observations at t
        prediction_at_t_next = model(torch.tensor(obs_t_next).float().to("cuda")).to("cpu")  # Use the model to predict an action using observations at t+1

        X = []
        y = []
        i = 0

        for obs_t, action, reward, _ in batch_vector:

            X.append(obs_t)

            # bellman optimality equation
            target = reward + opt.gamma * np.max(prediction_at_t_next[i].numpy())

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
        y_pred = model(torch.tensor(X).float().to("cuda"))
        loss_val = model.loss_fn(y_pred, torch.tensor(y).to("cuda"))

        loss.append(loss_val)
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
        self.fc1 = nn.Linear(obs_count, 100)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(100, 100)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(100, action_count)

        self.loss_fn = nn.SmoothL1Loss()  # Huber loss
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=opt.learning_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x


model = DQN(obs_count, action_count).to("cuda")

rewards = []
loss = []
epsilons = []
episodes = opt.num_episodes
episode_len = opt.episode_len
beta = opt.beta  # This is the epsilon decay rate
batch_size = opt.batch_size
memory = deque([], maxlen=2500)

for episode in range(episodes):
    environment = Environment(
        0.1,  # cloudiness
        0.5)  # energy_consumption
    obs_t = environment.get_state()
    total_reward = 0
    epsilon = 1 / (1 + beta * (episode / action_count))

    for ep_index in tqdm(range(episode_len)):
        with torch.no_grad():
            rand_num = np.random.random()
            if rand_num <= epsilon:
                action_index = random.randint(0, action_count-1)
            else:
                action = model(torch.tensor(obs_t).float().to("cuda")).to("cpu")
                action_index = np.argmax(action)

            # set actions
            if action_index == 0:
                heating = True
                cooling = False
            elif action_index == 1:
                heating = False
                cooling = True
            else:
                heating = False
                cooling = False

            environment.run(heating=heating, cooling=cooling, steps=1, output_format='none')
            obs_t_next = environment.get_state()
            reward = environment.calculate_reward(environment.greenhouse.temp, environment.H_temp, heating) # input current variables here
            total_reward += reward
            memory.append((obs_t, action_index, reward, obs_t_next))
            obs_t = obs_t_next

        if len(memory) > batch_size:
            loss = experience_replay(model, batch_size, memory, obs_count, action_count, 1, loss)

    rewards.append(total_reward)
    avg_reward = total_reward/episode_len
    print("\n Episode", episode+1, "of", opt.num_episodes, "| average reward: %.2f" % avg_reward,"\n")

plot(environment)

#torch.save(model, 'test01')
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from collections import deque
import random
from pylab import rcParams
from tqdm import tqdm

from options import Options
opt = Options()
from environment import Environment  #import environment simulation


def experience_replay(model, batch_size, gamma, memory, obs_count, action_count, epoch_count, loss):
    with torch.no_grad():
        batch = random.sample(memory, batch_size)  # sample a batch from the experience memory using batch_size
        batch_vector = np.array(batch, dtype=object)  # vectorise the batch

        obs_t = np.zeros(shape=(
            batch_size, obs_count))  # observation at t: create a numpy array of zeros with dimensions batch_size, state_count
        obs_t_next = np.zeros(shape=(
            batch_size, obs_count))  # observation at t+1: create a numpy array of zeros with dimensions batch_size, state_count

        for i in range(len(batch_vector)):  # loop through the batch collecting the obs at t and obs at t+1
            obs_t[i] = batch_vector[i, 0]  # store the observations at t in the relevant array
            obs_t_next[i] = batch_vector[i, 3]  # store the observations at t+1 in the relevant array

        prediction_at_t = model(torch.tensor(obs_t).float().to("cuda")).to("cpu")  # Use the model to predict an action using observations at t
        prediction_at_t_next = model(torch.tensor(obs_t_next).float().to("cuda")).to("cpu")  # Use the model to predict an action using observations at t+1

        # Create the features(X) and lables(y)
        X = []  # This is our feature vector, the most recent observation
        y = []  # This is our label calculated using the long term discounted reward
        i = 0

        for obs_t, action, reward, _ in batch_vector:  # get a row from our batch

            X.append(obs_t)  # append the most recent observation to X
            target = reward + gamma * np.max(prediction_at_t_next[i].numpy())
            # now we update the action value for the original state(observation) given the action that
            # was taken, and we use the target value as the update.
            prediction_at_t[i, action] = target  # <- One line of code here
            # the updated action values are used as the label
            y.append(
                prediction_at_t[i].numpy())  # <- One line of code here, remember the update will be used as the label to update
            # the ANN weights by backpropagating the mean squared error.

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

    heating_plot = list_window_averaging(win_len=50, list_to_avg=world.H_greenhouse_heatingThing)
    ventilation_plot = list_window_averaging(win_len=50, list_to_avg=world.H_greenhouse_coolingThing)

    plt.plot([energy * max(world.H_greenhouse_temp) for energy in heating_plot],
             label='Heatinh', linewidth='2', color="red")

    plt.plot([cooler*max(world.H_greenhouse_temp) for cooler in ventilation_plot], label = "coolz :D", linewidth ='2', color="black")
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

class MyModel(nn.Module):
    def __init__(self, obs_count, action_count):
        super(MyModel, self).__init__()
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


model = MyModel(obs_count, action_count).to("cuda")

rewards = []
loss = []
epsilons = []
episodes = opt.num_episodes
episode_len = opt.episode_len
gamma = opt.gamma  # This is the discount rate
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

    for ep_index in tqdm(range(episode_len), mininterval=0.1):
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
            loss = experience_replay(model, batch_size, gamma, memory, obs_count, action_count, 1, loss)

    rewards.append(total_reward)
    avg_reward = total_reward/episode_len
    print("\n Episode", episode, "| average reward: %.2f" % avg_reward,"\n")

plot(environment)

torch.save(model, 'test01')
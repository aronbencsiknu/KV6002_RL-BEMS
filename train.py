import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import torch
import torch.nn as nn
from collections import deque
import random
import time
from pylab import rcParams
from tqdm import tqdm

from environment import Environment

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

        #obs_t = torch.tensor(obs_t).float()
        #obs_t_next = torch.tensor(obs_t_next).float()


        prediction_at_t = model(torch.tensor(obs_t).float())  # Use the model to predict an action using observations at t
        prediction_at_t_next = model(torch.tensor(obs_t_next).float())  # Use the model to predict an action using observations at t+1

        # Create the features(X) and lables(y)
        X = []  # This is our feature vector, the most recent observation
        y = []  # This is our label calculated using the long term discounted reward
        i = 0

        for obs_t, action, reward, _, done in batch_vector:  # get a row from our batch

            X.append(obs_t)  # append the most recent observation to X
            if done:  # if the episode was over
                #target = reward.cpu().detach().numpy()  # the target value is just the reward
                target = reward  # the target value is just the reward
            else:  # otherwise
                # the target value is the discounted optimal reward (Bellman optimality equation)
                # Remember we use the max action_value from the state(observation) at time t+1
                target = reward + gamma * np.max(prediction_at_t_next[i].numpy())  # <- One line of code here
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
        y_pred = model(torch.tensor(X).float())
        loss_val = model.loss_fn(y_pred, torch.tensor(y).float())

        loss.append(loss_val)
        #print(loss_val.item())
        # Backward pass and optimization step
        model.optimizer.zero_grad()
        loss_val.backward()
        model.optimizer.step()

    return loss  # return the loss

def plot(world):
    rcParams['figure.figsize'] = 20, 5

    plt.plot(world.H_temp, label='Outside temperature', linewidth='10', color="blue")

    plt.plot([light * max(world.H_greenhouse_temp) for light in world.H_sunlight],
             label='Sunlight', linewidth='2', color="orange")

    plt.plot([energy / max(world.H_greenhouse_temp) for energy in world.H_greenhouse_energy_consumption],
             label='Consumed energy', linewidth='3', color="red")

    plt.plot(world.H_greenhouse_temp, label='Greenhouse temperature', linewidth='3', color="green")

    #lake below
    plt.plot([cooler/max(world.H_greenhouse_temp) for cooler in world.H_greenhouse_coolingThing], label = "coolz :D", linewidth ='2', color="black")
    # plt.figure(figsize=(10, 5))
    custom_ticks, custom_tick_names = world.get_custom_xcticks(world.H_temp)
    plt.xticks(custom_ticks, custom_tick_names)
    # custom_xticks = get_custom_xticks(len(world.H_greenhouse_temp))
    plt.legend()
    plt.show()

"""cartpole_env = gym.make('CartPole-v1')
observation = cartpole_env.reset()
obs_count = cartpole_env.observation_space.shape[0]
action_count = cartpole_env.action_space.n"""

environment = Environment(
    0.1,  # cloudiness
    0.5)  # energy_consumption
observation = environment.get_state()
obs_count = len(observation)
action_count = 2

heating = False
cooling = False

alpha = 0.001

class MyModel(nn.Module):
    def __init__(self, obs_count, action_count, alpha):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(obs_count, 50)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(50, 50)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(50, action_count)
        #self.lsm = nn.LogSoftmax(dim=1)

        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=alpha)

    def forward(self, x):
        x = x.to("cuda")
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        #x = self.lsm(x)
        return x.to("cpu")


model = MyModel(obs_count, action_count, alpha).to("cuda")

rewards = []  # store the rewards in a list for analysis only
loss = []  # store the losses in a list for analysis only
epsilons = []
episodes = 5
episode_len = 5000
gamma = 0.9  # This is the discount rate
beta = 0.3  # This is the epsilon decay rate
batch_size = 16  # The batch size for solving the IID problem
memory = deque([], maxlen=2500)  # The memory replay buffer set to a length of2500

# Please read through this code and try and understand what is happening, you should understand this
# if you have watched the videos but if you need any help ask the lab supervisor.
for episode in range(episodes):
    environment = Environment(
        0.1,  # cloudiness
        0.5)  # energy_consumption
    #obs_t = cartpole_env.reset()
    #obs_t = np.reshape(obs_t[0], [1, obs_count])
    obs_t = environment.get_state()
    total_reward = 0
    epsilon = 1 / (1 + beta * (episode / action_count))

    done = False
    for ep_index in tqdm(range(episode_len), desc ="Episode progress"):
        with torch.no_grad():
            rand_num = np.random.random()
            if rand_num <= epsilon:
                action_index = random.randint(0, 1)
            else:
                action = model(torch.tensor(obs_t).float())
                #action = np.argmax(action_values[0])
            #_, action_index = torch.topk(action,1)
                action_index = np.argmax(action)

            #obs_t_next, reward, done, _, info = cartpole_env.step(action)

            if action_index == 0:
                heating = True
                cooling = False
            else:
                heating = False
                cooling = True

            environment.run(heating, cooling, 1, 'none')
            #plot(environment)
            obs_t_next = environment.get_state()
            reward = environment.calculate_reward(environment.greenhouse.temp, environment.H_temp, heating) # input current variables here
            #obs_t_next = np.reshape(obs_t_next, [1, obs_count])
            total_reward += reward
            memory.append((obs_t, action_index, reward, obs_t_next, done))
            obs_t = obs_t_next

            if done:
                rewards.append(total_reward)
                epsilons.append(epsilon)
                print(f'episode: {episode}/{episodes}, score: {total_reward}, epsilon: {epsilon}')
                if total_reward>=1000:
                    print("break")
                    torch.save(model, 'test01')

        if len(memory) > batch_size:
            loss = experience_replay(model, batch_size, gamma, memory, obs_count, action_count, 1, loss)  ############################
            #print(ep_index)
        if ep_index >= episode_len:
            done = True
    rewards.append(total_reward)
    avg_reward = total_reward/episode_len
    print()
    print("Episode", episode, "| average reward: %.2f" % avg_reward)
    print()
plot(environment)

torch.save(model, 'test01')
import json
import pathlib
import numpy as np
import torch
import torch.nn as nn
from collections import deque
import random
from progress.bar import ShadyBar
from datetime import datetime
import time

from environment import Environment  # import environment simulation
from options import Options  # import options
from reward import Reward  # import reward mechanism for agent
from plot import Plot  # import environment plotting
from model import DQN

# from plotting import Plot
opt = Options()

# we can declare max/min temps here, but we can also change them later via a setter method in Reward
reward = Reward()

# we can change the desired plotting colours here
plotting = Plot()

environment = Environment(
    0.1,  # cloudiness
    0.5)  # energy_consumption

if not opt.pre_train and not opt.local_demo:
    import requests
    response = requests.get('http://127.0.0.1:3000/index.html',
                            headers={'Cache-Control': 'no-cache', 'Pragma': 'no-cache'})

observation = environment.get_state()
obs_count = len(observation)
action_count = 3

model = DQN(obs_count, action_count).to(opt.device)

loss_fn = nn.SmoothL1Loss()  # Huber loss
optimizer = torch.optim.AdamW(model.parameters(), lr=opt.learning_rate)  # AdamW optimizer


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
        batch = random.sample(memory, batch_size)  # get random batch from memory with size batch_size
        batch_vector = np.array(batch, dtype=object)

        # create np arrays with correct shapes
        obs_t = np.zeros(shape=(batch_size, obs_count))
        obs_t_next = np.zeros(shape=(batch_size, obs_count))

        for i in range(len(batch_vector)):
            obs_t[i] = batch_vector[i, 0]
            obs_t_next[i] = batch_vector[i, 3]

        # predict actions for time t and time t+1
        prediction_at_t = model(torch.tensor(obs_t).float().to(opt.device)).to("cpu")
        prediction_at_t_next = model(torch.tensor(obs_t_next).float().to(opt.device)).to("cpu")

        X = []  # data list
        y = []  # target list

        i = 0
        for obs_t, action, reward_value, _ in batch_vector:
            # append to data list
            X.append(obs_t)

            # bellman optimality equation
            target = reward_value + opt.gamma * np.max(prediction_at_t_next[i].numpy())

            # update action value
            prediction_at_t[i, action] = target

            # append to target list
            y.append(
                prediction_at_t[i].numpy())

            i += 1

        X = np.array(X).reshape(batch_size, obs_count)
        y = np.array(y)

        loss = []

    for epoch in range(epoch_count):
        # Forward pass
        y_pred = model(torch.tensor(X).float().to(opt.device))
        loss_val = loss_fn(y_pred, torch.tensor(y).to(opt.device))
        loss.append(loss_val.item())

        # Backward pass and optimization step
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

    return loss


def run_iter(obs_t, total_reward, epsilon):
    with torch.no_grad():
        rand_num = np.random.random()

        # explore
        if rand_num <= epsilon:
            action_index = random.randint(0, action_count - 1)

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
        reward_value = reward.calculate_reward(environment.greenhouse.temp, environment.H_temp,
                                               heating)  # input current variables here
        total_reward += reward_value

        # append to experience memory
        memory.append((obs_t, action_index, reward_value, obs_t_next))
        obs_t = obs_t_next

        return obs_t, total_reward


rewards = []
loss = []
epsilons = []
beta = opt.beta  # This is the epsilon decay rate
batch_size = opt.batch_size
memory = deque([], maxlen=2500)

if opt.pre_train:
    for episode in range(opt.num_episodes):
        environment = Environment(
            0.1,  # cloudiness
            0.5)  # energy_consumption
        obs_t = environment.get_state()
        total_reward = 0

        epsilon = 1 / (1 + opt.beta * (episode / action_count))
        epsilons.append(epsilon)

        bar_title = "Episode " + str(episode + 1) + " of " + str(opt.num_episodes)
        bar = ShadyBar(bar_title, max=opt.episode_len)
        for ep_index in range(opt.episode_len):

            if ep_index % 100 == 0:
                midpoint = random.randint(20, 25)

                """reward.update(max_temp=midpoint + random.randint(1, 4),
                              min_temp=midpoint - random.randint(1, 4),
                              crit_max_temp=midpoint + random.randint(4, 8),
                              crit_min_temp=midpoint - random.randint(4, 8),
                              max_crit_time=random.randint(30, 60),
                              max_allowed_temp_change=random.randint(1, 5))"""

            obs_t, total_reward = run_iter(obs_t, total_reward, epsilon)
            # train DQN and calculate loss
            if len(memory) > batch_size:
                loss = experience_replay(model, batch_size, memory, obs_count, opt.num_epochs)

            bar.next()  # update progress bar

        bar.finish()

        rewards.append(total_reward)
        avg_reward = total_reward / opt.episode_len
        print("\t - avg reward: %.4f" % avg_reward, "\n"
                                                    "\t - avg loss: %.4f" % np.mean(np.asarray(loss)), "\n"
                                                                                                       "\t - epsilon: %.4f" % epsilon,
              "\n")

    current_dateTime = datetime.now()
    path = pathlib.Path("trained_model")
    torch.save(model.state_dict(), path)
    plotting.plot(environment)

else:
    reward = Reward()
    current_dir = pathlib.Path(__file__).resolve()
    parent_dir = current_dir.parents[1]
    path = pathlib.Path(parent_dir / "trained_models/trained_model")

    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))

    environment = Environment(
        0.1,  # cloudiness
        0.5)  # energy_consumption
    if opt.local_demo:
        bar_title = "Progress"
        bar = ShadyBar(bar_title, max=opt.demo_len)
    for i in range(opt.demo_len):
        if not opt.local_demo:
            time.sleep(opt.demo_sleep)
            try:
                # LAKEvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
                with open(pathlib.Path(parent_dir / "public/json/gh1_settings.json"), 'r') as f:
                    data = json.load(f)
                # LAKE^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

                # LAKEvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
                reward.min_temp = int(data['minTemp'])
                reward.max_temp = int(data['maxTemp'])
                reward.max_allowed_temp_change = int(data['rateOfChange'])
                reward.crit_min_temp = int(data['critMinTemp'])
                reward.crit_max_temp = int(data['critMaxTemp'])
                reward.crit_time = int(data['maxTime'])
                # LAKE^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            except FileNotFoundError:
                print("settings not found")
                continue
        else:
            bar.next()  # update progress bar

        obs_t = environment.get_state()
        total_reward = 0

        #epsilon = 1 / (1 + opt.beta * (i / action_count))
        epsilon = 0
        epsilons.append(epsilon)

        obs_t, total_reward = run_iter(obs_t, total_reward, epsilon)
        # train DQN and calculate loss
        if len(memory) > batch_size:
            loss = experience_replay(model, batch_size, memory, obs_count, opt.num_epochs)

        rewards.append(total_reward)
        avg_reward = total_reward / opt.episode_len

        if not opt.local_demo:
            # LAKEvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
            avg_consumption = -5
            if environment.step > 60:
                last_60_energy = []
                last_60_energy = environment.H_greenhouse_heating[len(environment.H_greenhouse_heating) - 61: len(environment.H_greenhouse_heating) -1]
                avg_consumption =  np.mean(last_60_energy)
            with open(pathlib.Path(parent_dir / "public/json/gh1_obs.json"), 'w+') as f:
                json.dump([{"Greenhouse_temp": environment.greenhouse.temp, 
                            "Outside_temp": environment.temp,
                            "Time": 0,
                            "Ventilation": int(environment.greenhouse.ventilation),
                            "Heating":int(environment.greenhouse.heating),
                            "Average_consumption": avg_consumption}], f)

            # LAKE^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    if opt.local_demo:
        bar.finish()
        plotting.plot(environment)

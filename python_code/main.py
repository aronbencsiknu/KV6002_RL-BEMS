# %% imports
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
import argparse
import os
import wandb

from environment import Environment  # import environment simulation
from options import Options  # import options
from reward import Reward  # import reward mechanism for agent
from plot import Plot  # import environment plotting
from model import DQN  # import DQN model

# %% define variables

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-l", "--localdemo", help="increase output verbosity",
                    action="store_true")

parser.add_argument("-p", "--pretrain", help="increase output verbosity",
                    action="store_true")

parser.add_argument("-g",'--gindex', type=int)
args = parser.parse_args()

# from plotting import Plot
opt = Options()

# we can declare max/min temps here, but we can also change them later via a setter method in Reward
reward = Reward()

# we can change the desired plotting colours here
plotting = Plot()

# initialize environment simulation
environment = Environment(
    0.1,  # cloudiness
    0.5)  # energy_consumption


observation = environment.get_state()  # get initial observation of the environment
obs_count = len(observation)  # get the number of environment observations

# number of possible actions (heat on, vent closed; vent open, heat off; both off)
action_count = 3 
model = DQN(obs_count, action_count).to(opt.device)
loss_fn = opt.loss_fn
optimizer = torch.optim.AdamW(model.parameters(), lr=opt.learning_rate)  # AdamW optimizer
beta = opt.beta  # epsilon decay rate

# lists
rewards = []  # historical reward values
loss = []  # historical loss values
epsilons = []  # historical epsilon values
memory = deque([], maxlen=2500)  # experience memory


# %% function definitions
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

        #print(obs_t)
        # predict actions for time t and time t+1
        prediction_at_t = model(torch.tensor(obs_t).float().to(opt.device).float()).to("cpu")
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


def run_iter(obs_t, epsilon):
    """

    :param obs_t: observation at time t
    :param epsilon: current epsilon value
    :return: returns the observation at time t and the total reward
    """

    # explore
    if np.random.random() <= epsilon:
        action_index = random.randint(0, action_count - 1)

    # exploit
    else:
        with torch.no_grad():
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
    reward_value, r1, r2, r3 = reward.calculate_reward(environment.greenhouse.temp, environment.H_temp,
                                           heating)  # input current variables here
    

    # append to experience memory
    memory.append((obs_t, action_index, reward_value, obs_t_next))
    obs_t = obs_t_next

    return obs_t, reward_value, r1, r2, r3


# %% program run

# uhm, this does something with the GUI, I'm not sure, i didn't write this
if not args.pretrain and not args.localdemo:
    import requests
    response = requests.get('http://127.0.0.1:3000/index.html',
                            headers={'Cache-Control': 'no-cache', 'Pragma': 'no-cache'})

# run pre-training
if args.pretrain:

    # optional WandB logging
    if opt.wandb:
        key=opt.wandb_key
        wandb.login(key=key)
        wandb_group = "test"

        wandb.init(project="RL_GEMS", 
                group=wandb_group, entity="aronbencsik", 
                settings=wandb.Settings(start_method="thread"))
        
    for episode in range(opt.num_episodes):
        environment = Environment(
            0.1,  # cloudiness
            0.5)  # energy_consumption
        obs_t = environment.get_state()

        epsilon = 1 / (1 + opt.beta * (episode / action_count))
        epsilons.append(epsilon)

        bar_title = "Episode " + str(episode + 1) + " of " + str(opt.num_episodes)
        bar = ShadyBar(bar_title, max=opt.episode_len)

        total_reward = 0

        # wandb logging averages
        wandb_avg_r1 = 0
        wandb_avg_r2 = 0
        wandb_avg_r3 = 0
        wandb_avg_r_t = 0
        wandb_avg_l = 0

        for ep_index in range(opt.episode_len):
            
            #if ep_index % 500 == 0:
            if True == False:
                midpoint = random.randint(22, 23)

                # training DQN to deal with changing reward values
                reward.update(max_temp=midpoint + random.randint(1, 2),
                              min_temp=midpoint - random.randint(1, 2),
                              crit_max_temp=midpoint + random.randint(3, 4),
                              crit_min_temp=midpoint - random.randint(3, 4),
                              max_crit_time=random.randint(30, 60),
                              max_allowed_temp_change=random.randint(1, 5))

            obs_t, reward_value, r1, r2, r3 = run_iter(obs_t, epsilon)
            
            total_reward += reward_value

            # early stopping
            

            # train DQN and calculate loss
            if len(memory) > opt.batch_size:
                
                loss = experience_replay(model, opt.batch_size, memory, obs_count, opt.num_epochs)
                if opt.wandb:
                    wandb_avg_r1 += r1
                    wandb_avg_r2 += r2
                    wandb_avg_r3 += r3
                    wandb_avg_r_t += reward_value
                    wandb_avg_l += np.mean(np.asarray(loss))

                    if ep_index % 50 == 0:
                
                        wandb.log({"Loss": wandb_avg_l/50,
                        "Total reward": wandb_avg_r_t/50,
                        "Temp range reward": wandb_avg_r1/50,
                        "Energy reward": wandb_avg_r2/50,
                        "Temp change reward": wandb_avg_r3/50})

                        wandb_avg_r1 = 0
                        wandb_avg_r2 = 0
                        wandb_avg_r3 = 0
                        wandb_avg_r_t = 0
                        wandb_avg_l = 0


            bar.next()  # update progress bar

        bar.finish()
        avg_reward = total_reward / opt.episode_len
        print("\t - avg reward: %.4f" % avg_reward, "\n"
                                                    "\t - avg loss: %.4f" % np.mean(np.asarray(loss)), "\n"
                                                                                                       "\t - epsilon: %.4f" % epsilon,
              "\n")

    current_dateTime = datetime.now()
    current_dir = pathlib.Path(__file__).resolve()
    parent_dir = current_dir.parents[1]
    path = pathlib.Path(parent_dir / opt.path_to_model_from_root / opt.model_name_save)
    torch.save(model.state_dict(), path)
    plotting.plot(environment)

# run local or GUI demo
else:
    obs_t = environment.get_state()
    # load saved model
    current_dir = pathlib.Path(__file__).resolve()
    parent_dir = current_dir.parents[1]
    path = pathlib.Path(parent_dir / opt.path_to_model_from_root / opt.model_name_load)
    model.load_state_dict(torch.load(path, map_location=torch.device(opt.device)))
    
    # add progress bar for local demo
    if args.localdemo:
        bar_title = "Progress"
        bar = ShadyBar(bar_title, max=opt.demo_len)

    total_reward = 0

    for i in range(opt.demo_len):

        # receive data from GUI
        if not args.localdemo:
            time.sleep(opt.demo_sleep)
            try:
                path = pathlib.Path(os.path.join(parent_dir, "public", "json", "gh" + str(args.gindex) + "_settings.json"))
                # LAKEvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
                with open(path, 'r') as f:
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

        epsilon = 0
        epsilons.append(epsilon)

        obs_t, reward_value, _, _, _ = run_iter(obs_t, epsilon)

        total_reward += reward_value

        # train DQN and calculate loss
        if len(memory) > opt.batch_size:
            loss = experience_replay(model, opt.batch_size, memory, obs_count, opt.num_epochs)

        # send data to GUI
        if not args.localdemo:
            # LAKEvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
            avg_consumption = -5
            if environment.step > 60:
                last_60_energy = environment.H_greenhouse_heating[len(environment.H_greenhouse_heating) - 61: len(environment.H_greenhouse_heating) -1]
                avg_consumption = np.mean(last_60_energy)

            path = pathlib.Path(os.path.join(parent_dir, "public", "json", "gh" + str(args.gindex) + "_obs.json"))

            # format elapsed time
            day = "{:02d}".format(environment.day)
            hour = "{:02d}".format(environment.hour)
            minute = "{:02d}".format(environment.minute)
            with open(path, 'w+') as f:
                json.dump({"Greenhouse_temp": environment.greenhouse.temp, 
                            "Outside_temp": environment.temp,
                            "Time": str(day) +" : "+str(hour) +" : "+ str(minute),
                            "Ventilation": int(environment.greenhouse.ventilation),
                            "Heating": int(environment.greenhouse.heating),
                            "Average_consumption": avg_consumption}, f)

            # LAKE^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    # finish progress bar and plot environment for local demo
    if args.localdemo:
        avg_reward = total_reward / opt.demo_len
        print("\n\n avg reward: %.4f" % avg_reward)
        bar.finish()
        plotting.plot(environment)

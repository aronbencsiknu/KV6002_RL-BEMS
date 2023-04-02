# %% imports
import json
import pathlib
import numpy as np
import torch
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

# argument for local demo
parser.add_argument("-l", "--localdemo", help="run demo locally without connecting to the server",
                    action="store_true")

# argument for training
parser.add_argument("-p", "--pretrain", help="train a new model",
                    action="store_true")

# argument for wandb logging
parser.add_argument("-wb", "--wandb", help="log to weights&biases",
                    action="store_true")

# argument used during GUI demo to define the greenhouse indices
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

# initialize DQN
model = DQN(obs_count, action_count).to(opt.device)

if not args.pretrain:

    # load saved model
    current_dir = pathlib.Path(__file__).resolve()
    parent_dir = current_dir.parents[1]
    path = pathlib.Path(parent_dir / opt.path_to_model_from_root / opt.model_name_load)
    model.load_state_dict(torch.load(path, map_location=torch.device(opt.device)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.demo_learning_rate)  # AdamW optimizer

else:
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.learning_rate)  # AdamW optimizer
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=opt.num_episodes)  # learning rate scheduler

model.eval()

loss_fn = opt.loss_fn  # loss function
beta = opt.beta  # epsilon decay rate
memory = deque([], maxlen=2500)  # experience memory

# %% function definitions
def experience_replay(model, batch_size, memory, obs_count):
    """
    1. Selects random samples form the experience memory.
    2. Generates labels using the Bellman optimality equation
    3. Trains the DQN uaing the IIDD data

    --------
    :param model: DQN network
    :param batch_size: size of random samples taken from memory
    :param memory: experience memory
    :param obs_count: observation size of the environment
    :param epoch_count: epoch count for training the DQN
    :return:
    """
    
    batch = random.sample(memory, batch_size)  # get random batch from memory with size batch_size
    batch_vector = np.array(batch, dtype=object)

    # create np arrays with correct shapes
    obs_t = np.zeros(shape=(batch_size, obs_count))
    obs_t_next = np.zeros(shape=(batch_size, obs_count))

    # store observations at time t and time t+1 from the batch vector
    for i in range(len(batch_vector)):
        obs_t[i] = batch_vector[i, 0]
        obs_t_next[i] = batch_vector[i, 3]

    # predict actions for time t and time t+1
    with torch.no_grad():
        prediction_at_t = model(torch.tensor(obs_t).float().to(opt.device)).to("cpu")
        prediction_at_t_next = model(torch.tensor(obs_t_next).float().to(opt.device)).to("cpu")

    X = []  # training samples list
    y = []  # target list

    i = 0
    for obs_t, action, reward_value, _ in batch_vector:

        X.append(obs_t)  # append training sample list

        """ 
        bellman optimality equation:
        target = reward + discount_rate * argmax(DQN(obs_t+1))

        """
        target = reward_value + opt.gamma * np.max(prediction_at_t_next[i].numpy())

        # update action value
        prediction_at_t[i, action] = target

        # append to target list
        y.append(prediction_at_t[i].numpy())

        i += 1

    X = np.array(X).reshape(batch_size, obs_count)  # trainig samples 
    y = np.array(y)  # targets

    loss_avg = 0
    model.train()
    for epoch in range(opt.num_epochs):

        # Forward pass
        y_pred = model(torch.tensor(X).float().to(opt.device))
        loss_val = loss_fn(y_pred, torch.tensor(y).to(opt.device))
        loss_avg += loss_val.item()

        # Backward pass
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()
    
    model.eval()

    return loss_avg


def forward_pass(obs_t, epsilon):
    """
    1. Gets an action either by exploring or exploiting, depending on the current epsilon value.
    2. Advances simulation with the action.
    3. Calculates reward.
    4. Appends to experience memorx.

    --------
    :param obs_t: observation at time t
    :param epsilon: current epsilon value
    :return: returns the observation at time t and the total reward
    """

    # explore
    if np.random.random() < epsilon:
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
    environment.run(
        heating=heating, 
        cooling=ventilation, 
        steps=1, 
        output_format='none', 
        max_temp=reward.max_temp, 
        min_temp=reward.min_temp, 
        crit_max_temp=reward.crit_max_temp, 
        crit_min_temp=reward.crit_min_temp
    )

    

    # get environment state
    obs_t_next = environment.get_state()

    # calculate reward
    reward_value, r1, r2, r3 = reward.calculate_reward(
        environment.greenhouse.temp, 
        environment.H_temp,
        heating
    )

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
    if args.wandb:
        key=opt.wandb_key
        wandb.login(key=key)
        wandb_group = "demo"

        wandb.init(project="RL_GEMS", 
                group=wandb_group, entity="aronbencsik", 
                settings=wandb.Settings(start_method="thread"))
        
    for episode in range(opt.num_episodes):
        environment = Environment(
            0.1,  # cloudiness
            0.5)  # energy_consumption
        obs_t = environment.get_state()

        epsilon = 1 / (1 + opt.beta * (episode / action_count))

        bar_title = "Episode " + str(episode + 1) + " of " + str(opt.num_episodes)
        bar = ShadyBar(bar_title, max=opt.episode_len)

        total_reward = 0
        loss_avg = 0

        # wandb logging averages
        wandb_avg_r1 = 0
        wandb_avg_r2 = 0
        wandb_avg_r3 = 0
        wandb_avg_r_t = 0
        wandb_avg_l = 0

        for ep_index in range(opt.episode_len):
            
            if ep_index % 500 == 0:
                midpoint = random.randint(22, 23)

                # training DQN to deal with changing reward values
                reward.update(max_temp=midpoint + random.randint(1, 2),
                              min_temp=midpoint - random.randint(1, 2),
                              crit_max_temp=midpoint + random.randint(3, 4),
                              crit_min_temp=midpoint - random.randint(3, 4),
                              max_crit_time=random.randint(30, 60),
                              max_allowed_temp_change=random.randint(1, 5))
            
            # run forward pass and advance simulation
            obs_t, reward_value, r1, r2, r3 = forward_pass(obs_t, epsilon)
            
            total_reward += reward_value
            

            # train DQN and calculate loss
            if len(memory) > opt.batch_size:
                
                loss= experience_replay(model, opt.batch_size, memory, obs_count)
                loss_avg += loss

                if args.wandb:

                    # increment WandB averages
                    wandb_avg_r1 += r1
                    wandb_avg_r2 += r2
                    wandb_avg_r3 += r3
                    wandb_avg_r_t += reward_value
                    wandb_avg_l += loss/opt.num_epochs

                    if ep_index % opt.wandb_logging_freq == 0:
                
                        # log values at frequency {opt.wandb_logging_freq}
                        wandb.log({
                            "Loss": wandb_avg_l/opt.wandb_logging_freq,
                            "Total reward": wandb_avg_r_t/opt.wandb_logging_freq,
                            "Temp range reward": wandb_avg_r1/opt.wandb_logging_freq,
                            "Energy reward": wandb_avg_r2/opt.wandb_logging_freq,
                            "Temp change reward": wandb_avg_r3/opt.wandb_logging_freq,
                            "Epsilon": epsilon,
                            "Learning rate": optimizer.param_groups[0]["lr"]
                        })

                        # reset WandB averages
                        wandb_avg_r1 = 0
                        wandb_avg_r2 = 0
                        wandb_avg_r3 = 0
                        wandb_avg_r_t = 0
                        wandb_avg_l = 0


            bar.next()  # update progress bar

        bar.finish()
        avg_reward = total_reward / opt.episode_len
        scheduler.step()

        print("\t - Avg. reward: %.4f" % avg_reward, "\n"
              "\t - Avg. loss: %.4f" % (loss_avg / opt.episode_len), "\n"
              "\t - Epsilon: %.4f" % epsilon, "\n"
              "\t - Learning rate: %.6f" % optimizer.param_groups[0]["lr"], 
              "\n")

    current_dateTime = datetime.now()
    current_dir = pathlib.Path(__file__).resolve()
    parent_dir = current_dir.parents[1]
    path = pathlib.Path(parent_dir / opt.path_to_model_from_root / opt.model_name_save)
    torch.save(model.state_dict(), path)
    if args.wandb:
        wandb.finish()
    plotting.plot(environment)

# run local or GUI demo
else:
    
    obs_t = environment.get_state()
    epsilon = 0
    total_reward = 0
    loss_avg = 0

    if args.localdemo:
        print("Model loaded.")
        print("Model name: ", opt.model_name_load,"\n")
        demo_len = opt.local_demo_len
        bar_title = "Progress"
        bar = ShadyBar(bar_title, max=opt.local_demo_len)
    else:
        demo_len = opt.gui_demo_len

    for i in range(demo_len):

        # if demo-ing with GUI, read and write to JSON file
        if not args.localdemo:
            time.sleep(opt.demo_sleep)
            # LAKEvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
            
            # default value is -5 when there is not enough data yet
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
                            "Time": str(day) +":"+str(hour) +":"+ str(minute),
                            "Ventilation": int(environment.greenhouse.ventilation),
                            "Heating": int(environment.greenhouse.heating),
                            "Average_consumption": avg_consumption}, f)

            # LAKE^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
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
        
        # if demo-ing locally, just update the progressbar
        else:
            bar.next()  # update progress bar

        # run forward pass and advance simulation
        obs_t, reward_value, r1, r2, r3 = forward_pass(obs_t, epsilon)
        total_reward += reward_value
        
        if len(memory) > opt.batch_size:
            # train DQN and calculate loss
            loss_avg += experience_replay(model, opt.batch_size, memory, obs_count)

    # finish progress bar and plot environment for local demo
    if args.localdemo:
        avg_reward = total_reward / opt.local_demo_len
        print("\n\n avg reward: %.4f" % avg_reward)
        bar.finish()
        plotting.plot(environment)

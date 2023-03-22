import pathlib

from environment import Environment  # import environment simulation
from plot import Plot  # import environment plotting
from model import DQN
from options import Options  # import options
import torch
import numpy as np

# from plotting import Plot
opt = Options()

# we can change the desired plotting colours here
plotting = Plot()

environment = Environment(
    0.1,  # cloudiness
    0.5)  # energy_consumption
obs_t = environment.get_state()
obs_count = len(obs_t)
action_count = 3

model = DQN(obs_count, action_count).to(opt.device)

path = pathlib.Path("saved_models/"+"2023_3_22_12_5_37")  # replace - with model name

model.load_state_dict(torch.load(path, map_location=opt.device))

for i in range(5000):
    with torch.no_grad():
        action = model(torch.tensor(obs_t).float().to(opt.device)).to("cpu")
        action_index = np.argmax(action)

        if action_index == 0:
            heating = True
            ventilation = False
        elif action_index == 1:
            heating = False
            ventilation = True
        else:
            heating = False
            ventilation = False

        environment.run(heating=heating, cooling=ventilation, steps=1, output_format='none')
        obs_t = environment.get_state()

plotting.plot(environment)

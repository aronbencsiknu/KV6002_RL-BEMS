import numpy as np
import pathlib
import torch
from progress.bar import ShadyBar
from environment import Environment  # import environment simulation
from environment import Environment  # import environment simulation
from options import Options  # import options
from model import DQN  # import DQN model

# from plotting import Plot
opt = Options()

class manuelControl:
    def __init__(self, max_temp=25, min_temp=20, crit_max_temp=27, crit_min_temp=17, max_crit_time=70, max_allowed_temp_change=1):

        self.max_temp = max_temp
        self.min_temp = min_temp
        self.crit_max_temp = crit_max_temp
        self.crit_min_temp = crit_min_temp
        self.max_crit_time = max_crit_time
        self.max_allowed_temp_change = max_allowed_temp_change
        self.current_crit_time = 0
                                 # energy_consumption
        self.environment = Environment(
                                0.1,  # cloudiness
                                0.5) 
        self.avg60arr_m = []
        self.avg60arr_a = []
        self.daily_avgs_m = 0
        self.daily_avgs_a = 0 #initialise variables for compare 

    def Manuel(self, indoor_temp, indoor_temp_history):
        temp_change = 0
        try:
            prev_indoor_temp = indoor_temp_history[len(indoor_temp_history) - 2]  # get second last temp value
            temp_change = abs(indoor_temp - prev_indoor_temp)
        except:
            print("index not in range")
        else:
            temp_change = 0
        
       

        ventilation = False
        heating = False 
        temp_midpoint = (self.max_temp + self.min_temp) / 2  # midpoint
        
        if indoor_temp < self.min_temp and temp_change < self.max_allowed_temp_change:
            heating = True
            ventilation = False
        elif indoor_temp < self.min_temp and temp_change > self.max_allowed_temp_change:
            heating = False
            ventilation = False
        elif indoor_temp > self.max_temp:
            heating = False
            ventilation = True
        elif self.min_temp <= indoor_temp <= self.max_temp and (abs(indoor_temp - temp_midpoint) <= 0.5 * abs(self.max_temp - temp_midpoint) or abs(indoor_temp - temp_midpoint) <= 0.5 * abs(temp_midpoint - self.min_temp)): #switch off when temp is halfway between min/max temp and mid point
            heating = False
            ventilation = False

        return heating, ventilation
    
    def energyConsump(self,environment,avg60arr_m,environment_2,avg60arr_a):
        avg_consumption = -5
        if environment.step % 60 == 0 and environment.step > 59:
            last_60_energy = environment.H_greenhouse_heating[-61:-1]
            avg_consumption = np.mean(last_60_energy)
            avg60arr_m.append(avg_consumption)

        avg_consumption_auto = -5
        if environment_2.step % 60 == 0 and environment_2.step > 59:
            last_60_energy_auto = environment_2.H_greenhouse_heating[-61:-1]
            avg_consumption_auto = np.mean(last_60_energy_auto)
            avg60arr_a.append(avg_consumption_auto)

        if environment.step % 1440 == 0 and environment.step > 1:    
            daily_avgs_a = np.average(avg60arr_a)
            daily_avgs_m = np.average(avg60arr_m)
            if daily_avgs_m > daily_avgs_a:
                greaterEff = "Rule-Based Model"
            elif daily_avgs_m < daily_avgs_a:
                greaterEff = "Reinforced Learning Model"
            else:
                greaterEff = "Even Efficiency"
            with open('efficiency.txt', 'a') as file:
                file.write('Manual Daily Average: ' + str(daily_avgs_a) + '\n')
                file.write('AI Daily Average: ' + str(daily_avgs_m) + '\n')
                file.write(str(greaterEff) + " had greater efficiency" + '\n')

    def forward(self, g_temp, env_temp_h, environment):
        heating, ventilation = self.Manuel(g_temp, env_temp_h)
        self.environment.run(heating=heating, cooling=ventilation, steps=1, output_format='none')
        self.energyConsump(self.environment,self.avg60arr_m,environment,self.avg60arr_a)

        return heating, ventilation
    
# initialize environment simulation
environment_ai = Environment(
    0.1,  # cloudiness
    0.5)  # energy_consumption
environment_rule = Environment(
    0.1,  # cloudiness
    0.5)  # energy_consumption

manuelModel = manuelControl()
observation = environment_ai.get_state()  # get initial observation of the environment
obs_count = len(observation)  # get the number of environment observations

# number of possible actions (heat on, vent closed; vent open, heat off; both off)
action_count = 3 

model = DQN(obs_count, action_count).to(opt.device)    
# load saved model
current_dir = pathlib.Path(__file__).resolve()
parent_dir = current_dir.parents[1]
path = pathlib.Path(parent_dir / opt.path_to_model_from_root / opt.model_name_load)
model.load_state_dict(torch.load(path, map_location=torch.device(opt.device)))
print("Model loaded.")
print("Model name: ", opt.model_name_load,"\n")
demo_len = opt.local_demo_len
bar_title = "Progress"
bar = ShadyBar(bar_title, max=opt.local_demo_len)

avg60arr_m = []
avg60arr_a = []

for i in range(opt.local_demo_len):
    obs_t_ai = environment_ai.get_state()
    obs_rule = environment_rule.get_state()
    with torch.no_grad():
            action = model(torch.tensor(obs_t_ai).float().to(opt.device)).to("cpu")
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
    environment_ai.run(
        heating=heating, 
        cooling=ventilation, 
        steps=1, 
        output_format='none', 
        max_temp=0, 
        min_temp=0, 
        crit_max_temp=0, 
        crit_min_temp=0
    )
    heating, ventilation = manuelModel.Manuel(environment_rule.greenhouse.temp, environment_rule.H_temp)

    environment_rule.run(
        heating=heating, 
        cooling=ventilation, 
        steps=1, 
        output_format='none', 
        max_temp=0, 
        min_temp=0, 
        crit_max_temp=0, 
        crit_min_temp=0
    )

    manuelModel.energyConsump(environment_rule,avg60arr_m,environment_ai,avg60arr_a)

    bar.next()  # update progress bar

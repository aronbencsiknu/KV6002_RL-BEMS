# %%
import random
import math
import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams


# %%
"""def get_custom_xcticks(historical_data):
    ticks = []
    tick_names = []
    last_tick_name = 0
    for step in range(len(historical_data)):
        if (step % 60 == 0):
            ticks.append(step)
            tick_names.append(last_tick_name)
            last_tick_name += 1
            if (last_tick_name >= 24):
                last_tick_name = 0
    return ticks, tick_names"""


# %%
class Greenhouse():
    def __init__(self, energy_consumption):
        # natural status
        self.temp = 10
        self.heating_from_sun_rate = 0.1  # how fast the greenhouse heats up from sun
        self.temperature_diffusion_rate = 0.002  # how fast the greenhouse is affected by outside temperature
        # utilities
        self.heating = False
        self.cooling = False
        self.energy_consumption = energy_consumption
        self.energy_consumed = 0
        self.coolzed_down = 0 #lake


# %%
class Environment:
    def __init__(self, cloudiness, energy_consumption):
        # time
        self.step = 0
        self.day = 0
        self.hour = 0
        self.minute = 0
        # environment data
        self.local_temps = [6, 5, 5, 4, 3, 4, 5, 7, 9, 11, 12, 13, 14, 16, 14, 13, 12, 11, 10, 9, 8, 7, 7, 6]
        self.sunlight_intensities = [0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.4, 0.5, 0.6, 0.7, 0.9, 1, 1, 1, 1, 1, 0.9,
                                     0.8, 0.7, 0.6, 0.4, 0.0, 0.0]
        # environment
        self.temp = self.local_temps[0]
        self.sunlight = self.sunlight_intensities[0]
        self.cloudiness = cloudiness
        self.clouds = False
        self.cloud_remaining_life = 0
        # greenhouse (will be list in the future)
        self.greenhouse = Greenhouse(energy_consumption)
        # data history
        self.H_temp = []
        self.H_sunlight = []
        self.H_greenhouse_temp = []
        self.H_greenhouse_energy_consumption = []
        self.H_greenhouse_coolingThing = [] #lake
        self.H_hour = []
        self.H_minute = []

    def get_time_from_step(self):
        day = math.floor(self.step / 1440)
        hour = math.floor((self.step - (1440 * day)) / 60)
        minute = self.step % 60
        return day, hour, minute

    # -----------------------------------------------------------------------------------
    def show_stats(self, mode):  # mode = hourly / all / none
        if (mode == 'all'):
            self.show()
        if (mode == 'hourly'):
            if (self.step % 60 == 0):
                self.show()

    # -----------------------------------------------------------------------------------
    def show(self):
        print('----------------')
        print('STEP: ', self.step, '/// DAY: ', self.day)
        print('TIME: ', self.hour, ':', self.minute)
        print('Outside temperature: ', self.temp, '/// Sunlight intensity: ', self.sunlight)
        print('Greenhouse temperature: ', self.greenhouse.temp)
        print('Heating: ', str(self.greenhouse.heating))

    # -----------------------------------------------------------------------------------
    def shuffle_temps(self):
        change_amount = random.randint(5, 10)
        for change in range(change_amount):
            index = random.randrange(len(self.local_temps))
            self.local_temps[index] = random.randint(self.local_temps[index] - 3, self.local_temps[index] + 3)

    # -----------------------------------------------------------------------------------
    def simulate_environment(self):
        # CLOUD SPAWNING
        if random.random() < self.cloudiness:
            self.clouds = True
            self.cloud_remaining_life = random.uniform(self.cloudiness * 10, self.cloudiness * 100)

        # TEMPERATURE
        if self.temp < self.local_temps[self.hour]:
            self.temp += random.uniform(0.001, 0.04)
        elif self.temp > self.local_temps[self.hour]:
            self.temp -= random.uniform(0.001, 0.04)

        # randomizing some daily temperatures every day
        if self.step % 1440 == 0:
            self.shuffle_temps()

        # SUNLIGHT
        if self.clouds:
            if self.sunlight > 0:
                self.sunlight -= random.uniform(0.001, 0.04)
            self.cloud_remaining_life -= 1
            if self.cloud_remaining_life <= 0:
                self.clouds = False
        else:
            if self.sunlight < self.sunlight_intensities[self.hour]:
                self.sunlight += random.uniform(0.001, 0.04)
            elif self.sunlight > self.sunlight_intensities[self.hour]:
                self.sunlight -= random.uniform(0.001, 0.04)

        # normalize sunlight
        if self.sunlight > 1:
            self.sunlight = 1
        if self.sunlight < 0:
            self.sunlight = 0
        return self

    # -----------------------------------------------------------------------------------
    def simulate_greenhouse(self, heating, cooling):
        # natural change
        self.greenhouse.temp += self.sunlight * self.greenhouse.heating_from_sun_rate  # greenhouse temp increase from sun
        distance = abs(self.temp - self.greenhouse.temp)  # difference between greenhouse and outside temp
        change = distance * self.greenhouse.temperature_diffusion_rate  # how much the temperature of greenhouse should change
        if self.greenhouse.temp > self.temp:
            self.greenhouse.temp -= change
        elif self.greenhouse.temp < self.temp:
            self.greenhouse.temp += change

        # heating controls
        # overwritten by RL output
        """if self.greenhouse.temp < 20:
            self.greenhouse.heating = True
        else:
            self.greenhouse.heating = False"""

        self.greenhouse.heating = heating
        self.greenhouse.cooling = cooling

        # change from heating
        if self.greenhouse.heating:
            self.greenhouse.temp += 0.1
            self.greenhouse.energy_consumed += self.greenhouse.energy_consumption

        if self.greenhouse.cooling:
            self.greenhouse.temp -= 0.1
            self.greenhouse.coolzed_down += 0.1 #lake

    # -----------------------------------------------------------------------------------
    def make_history(self):
        self.H_temp.append(self.temp)
        self.H_sunlight.append(self.sunlight)
        self.H_greenhouse_temp.append(self.greenhouse.temp)
        self.H_greenhouse_energy_consumption.append(self.greenhouse.energy_consumed)
        self.H_hour = self.hour
        self.H_minute = self.minute
        self.H_greenhouse_coolingThing.append(self.greenhouse.coolzed_down) #lake

    # -----------------------------------------------------------------------------------
    def get_state(self):
        data = []
        #data.append(self.step)
        #data.append(self.day)
        #data.append(self.hour)
        #data.append(self.minute)
        data.append(self.temp)
        #data.append(self.sunlight)
        data.append(self.greenhouse.temp)
        #data.append(self.greenhouse.heating)
        #data.append(self.greenhouse.cooling)
        #data.append(self.greenhouse.energy_consumption)
        return data

    def get_custom_xcticks(self, historical_data):
        ticks = []
        tick_names = []
        last_tick_name = 0
        for step in range(len(historical_data)):
            if (step % 60 == 0):
                ticks.append(step)
                tick_names.append(last_tick_name)
                last_tick_name += 1
                if (last_tick_name >= 24):
                    last_tick_name = 0
        return ticks, tick_names

    ######################################################################################################################################

    def calculate_reward(self, indoor_temp, indoor_temp_history, energy_consumption):

        # reward weighing
        r1_w = 1.0
        r2_w = 1.0
        r3_w = 1.0

        #!!placeholder values!!
        max_temp = 20
        min_temp = 15
        temp_midpoint = (max_temp + min_temp) / 2  # midpoint

        # !!placeholder values!!
        max_allowed_temp_change = 5  # degrees/minute
        prev_indoor_temp = indoor_temp_history[len(indoor_temp_history) - 2]  # get second last temp value
        temp_change = abs(indoor_temp - prev_indoor_temp)

        # Calculate reward for indoor temperature control
        if min_temp <= indoor_temp <= max_temp:
            r1 = 1.0
        else:
            r1 = -abs(indoor_temp - temp_midpoint)

        # Calculate reward for energy savings
        if energy_consumption:
            r2 = 0.0
        else:
            r2 = 1.0

        if max_allowed_temp_change < (temp_change + 0.1) and max_allowed_temp_change < (temp_change - 0.1):
            r3 = 1.0
        else:
            r3 = -abs(max_allowed_temp_change - temp_change)

        # Calculate total reward

        r1 *= r1_w
        r2 *= r2_w
        r3 *= r3_w

        total_reward = r1 + r2 + r3

        #total_reward = -math.log(abs(temp_midpoint - indoor_temp))

        return total_reward

    ######################################################################################################################################
    def run(self, heating, cooling, steps, output_format):
        for step in range(steps):
            self.step += 1

            self.day, self.hour, self.minute = self.get_time_from_step()  # get day, hour and minutes

            self.simulate_environment()  # update world temperature and sunlight

            self.simulate_greenhouse(heating, cooling)  # simulate the greenhouse

            self.make_history()  # adds current data to memory (for future visualisation)

            self.show_stats(
                output_format)  # output to console. 'hourly' = each hour; 'all' = every step (every minute); 'none' = dont print


# %%
# PARAMETERS
# cloudiness: How cloudy it is. expects values 0-1. Determines how often clouds obstruct the sky and how long they last
# 0 = never 1 = always.
#
# WIP: this may be done so that cloudiness updates itself
# energy_consumption: How much energy per tick when heating (idk what this number resembles tho)
"""world = Environment(
    0.1,  # cloudiness
    0.5)  # energy_consumption"""

##############################################################################################################################################
# PARAMETERS
# duration: One step is equal to one minute
# verbose: 'hourly' = each hour; 'all' = every step (every minute); 'none' = dont print numerical data
# simulation works the same despite what gets printed
# NOTES
# running world.run(...) multiple times will continue the simulation from the last step. initialize world = Environment() to restart
"""world.run(
    6000,  # duration
    'none')  # verbose"""
# %%

"""rcParams['figure.figsize'] = 20, 5

plt.plot(world.H_temp, label='Outside temperature', linewidth='10', color="blue")

plt.plot([light * max(world.H_greenhouse_temp) for light in world.H_sunlight],
         label='Sunlight', linewidth='2', color="orange")

plt.plot([energy / max(world.H_greenhouse_temp) for energy in world.H_greenhouse_energy_consumption],
         label='Consumed energy', linewidth='5', color="red")

plt.plot(world.H_greenhouse_temp, label='Greenhouse temperature', linewidth='5', color="green")

# plt.figure(figsize=(10, 5))
custom_ticks, custom_tick_names = get_custom_xcticks(world.H_temp)
plt.xticks(custom_ticks, custom_tick_names)
# custom_xticks = get_custom_xticks(len(world.H_greenhouse_temp))
plt.legend()
plt.show()"""
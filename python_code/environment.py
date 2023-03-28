# %%
import random
import math
import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams

# %%
class Greenhouse():
    def __init__(self, energy_consumption):
        # natural status
        self.temp = 10
        self.heating_from_sun_rate = 0.1  # how fast the greenhouse heats up from sun
        self.temperature_diffusion_rate = 0.002  # how fast the greenhouse is affected by outside temperature
        # utilities
        self.heating = False
        self.ventilation = False
        self.energy_consumption = energy_consumption
        self.energy_consumed = 0

# %%
class Environment:
    def __init__(self, cloudiness, energy_consumption):
        # time
        self.step = 0
        self.day = 0
        self.hour = 0
        self.minute = 0
        # environment data
        self.local_temps = [6, 5, 5, 4, 3, 4, 5, 7, 9, 11, 12, 13, 14, 16, 14, 13, 12, 11, 10, 9, 8, 7, 7, 6, 6]
        self.sunlight_intensities = [0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.4, 0.5, 0.6, 0.7, 0.9, 1, 1, 1, 1, 1, 0.9,
                                     0.8, 0.7, 0.6, 0.4, 0.0, 0.0, 0.0]
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
        self.H_greenhouse_ventilation = [] #lake
        self.H_greenhouse_heating = []
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
        self.greenhouse.heating = heating
        self.greenhouse.ventilation = cooling

        # change from heating
        if self.greenhouse.heating:
            self.greenhouse.temp += 0.1
            self.greenhouse.energy_consumed += self.greenhouse.energy_consumption

        if self.greenhouse.ventilation:
            self.greenhouse.temp -= 0.15

    # -----------------------------------------------------------------------------------
    def make_history(self):
        self.H_temp.append(self.temp)
        self.H_sunlight.append(self.sunlight)
        self.H_greenhouse_temp.append(self.greenhouse.temp)
        self.H_greenhouse_energy_consumption.append(self.greenhouse.energy_consumed)
        self.H_hour = self.hour
        self.H_minute = self.minute
        self.H_greenhouse_ventilation.append(int(self.greenhouse.ventilation)) #lake
        self.H_greenhouse_heating.append(int(self.greenhouse.heating)) #lake

    # -----------------------------------------------------------------------------------
    def get_state(self):
        """
        collects relevant state information
        :return: state list of env at time t
        """

        # normalizing values for more stable learning process
        # normalizing values as: value' = (value - min) / (max - min)
        norm_hour = (self.hour - 0) / (23 - 0)
        norm_temp = (self.temp - 0) / (40 - 0)
        norm_greenhouse_temp = (self.greenhouse.temp - 0) / (40 - 0)
        norm_forecast_temp = (self.local_temps[self.hour + 1] - 0) / (40 - 0)

        # append data to observation array
        data = [norm_hour, norm_temp, self.sunlight, norm_greenhouse_temp, int(self.greenhouse.heating),
                int(self.greenhouse.ventilation), norm_forecast_temp, self.sunlight_intensities[self.hour + 1]]

        return data

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

plt.plotting(world.H_temp, label='Outside temperature', linewidth='10', color="blue")

plt.plotting([light * max(world.H_greenhouse_temp) for light in world.H_sunlight],
         label='Sunlight', linewidth='2', color="orange")

plt.plotting([energy / max(world.H_greenhouse_temp) for energy in world.H_greenhouse_energy_consumption],
         label='Consumed energy', linewidth='5', color="red")

plt.plotting(world.H_greenhouse_temp, label='Greenhouse temperature', linewidth='5', color="green")

# plt.figure(figsize=(10, 5))
custom_ticks, custom_tick_names = get_custom_xcticks(world.H_temp)
plt.xticks(custom_ticks, custom_tick_names)
# custom_xticks = get_custom_xticks(len(world.H_greenhouse_temp))
plt.legend()
plt.show()"""
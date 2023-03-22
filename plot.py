import matplotlib.pyplot as plt
from pylab import rcParams
import numpy as np


class Plot:
    def __init__(self):
        self.temp_c = "skyblue"
        self.sunlight_c = "khaki"
        self.greenhouse_temp_c = "mediumspringgreen"
        self.heating_c = "red"
        self.ventilation_c = "blue"

    def plot(self, world):

        rcParams['figure.figsize'] = 20, 5

        plt.plot(world.H_temp, label='Outside temperature', linewidth='2', color=self.temp_c, linestyle = ':')

        plt.plot([light * max(world.H_greenhouse_temp) for light in world.H_sunlight],
                 label='Sunlight', linewidth='2', color=self.sunlight_c)

        plt.plot(world.H_greenhouse_temp, label='Greenhouse temperature', linewidth='2', color=self.greenhouse_temp_c, linestyle = ':')

        # lake below

        heating_plot = self.list_window_averaging(win_len=50, list_to_avg=world.H_greenhouse_heating)
        ventilation_plot = self.list_window_averaging(win_len=50, list_to_avg=world.H_greenhouse_ventilation)

        plt.plot([energy * max(world.H_greenhouse_temp) for energy in heating_plot],
                 label='Heating', linewidth='2', color=self.heating_c)

        plt.plot([cooler * max(world.H_greenhouse_temp) for cooler in ventilation_plot], label="Ventilation",
                 linewidth='2',
                 color=self.ventilation_c)

        custom_ticks, custom_tick_names = self.get_custom_xcticks(world.H_temp)
        plt.xticks(custom_ticks, custom_tick_names)

        plt.legend()
        plt.show()

    def list_window_averaging(self, win_len, list_to_avg):
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

    def get_custom_xcticks(self, historical_data):
        ticks = []
        tick_names = []
        last_tick_name = 0
        for step in range(len(historical_data)):
            if step % 60 == 0:
                ticks.append(step)
                tick_names.append(last_tick_name)
                last_tick_name += 1
                if last_tick_name >= 24:
                    last_tick_name = 0
        return ticks, tick_names

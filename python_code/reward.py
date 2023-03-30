import wandb
class Reward:
    def __init__(self, max_temp=25, min_temp=20, crit_max_temp=27, crit_min_temp=17, max_crit_time=70, max_allowed_temp_change=1):

        self.max_temp = max_temp
        self.min_temp = min_temp
        self.crit_max_temp = crit_max_temp
        self.crit_min_temp = crit_min_temp
        self.max_crit_time = max_crit_time
        self.max_allowed_temp_change = max_allowed_temp_change
        self.current_crit_time = 0
    
    def update(self, max_temp, min_temp, crit_max_temp, crit_min_temp, max_crit_time, max_allowed_temp_change):
        self.max_temp = max_temp
        self.min_temp = min_temp
        self.crit_max_temp = crit_max_temp
        self.crit_min_temp = crit_min_temp
        self.max_crit_time = max_crit_time
        self.max_allowed_temp_change = max_allowed_temp_change

    def calculate_reward(self, indoor_temp, indoor_temp_history, heating):

        temp_midpoint = (self.max_temp + self.min_temp) / 2  # midpoint

        prev_indoor_temp = indoor_temp_history[len(indoor_temp_history) - 2]  # get second last temp value
        temp_change = abs(indoor_temp - prev_indoor_temp)

        # Calculate reward for indoor temperature control
        if self.min_temp <= indoor_temp <= self.max_temp:
            r1 = 0.0
            if self.current_crit_time > 0:
                self.current_crit_time -= 1

        elif self.crit_min_temp <= indoor_temp <= self.crit_max_temp and self.current_crit_time < self.max_crit_time:
            r1 = 0.0
            self.current_crit_time += 1

        else:
            r1 = -abs(indoor_temp - temp_midpoint)

        # Calculate reward for energy savings
        if heating:
            r2 = -1.0
        else:
            r2 = 0.0

        if self.max_allowed_temp_change < temp_change:
            r3 = 0.0
        else:
            r3 = -abs(self.max_allowed_temp_change - temp_change)

        # reward weighing
        r1_w = 1.0  # temperature range
        r2_w = 1.1  # energy
        r3_w = 1.0  # temp change

        # Calculate total reward
        r1_weighed = r1 * r1_w
        r2_weighed = r2 * r2_w
        r3_weighed = r3 * r3_w

        total_reward = r1_weighed + r2_weighed + r3_weighed

        return total_reward, r1, r2, r3


class Reward:
    def __init__(self, max_temp=25, min_temp=20, crit_max_temp=27, crit_min_temp=23, crit_time=30, max_allowed_temp_change=1):
        self.max_temp = max_temp
        self.min_temp = min_temp
        self.crit_max_temp = crit_max_temp
        self.crit_min_temp = crit_min_temp
        self.crit_time = crit_time
        self.max_allowed_temp_change = max_allowed_temp_change
        self.crit_time_actual = 0
        self.cooldown_time = 0
    
    def update(self, max_temp, min_temp, crit_max_temp, crit_min_temp, crit_time):
        self.max_temp = max_temp
        self.min_temp = min_temp
        self.crit_max_temp = crit_max_temp
        self.crit_min_temp = crit_min_temp
        self.crit_time = crit_time

    def calculate_reward(self, indoor_temp, indoor_temp_history, heating):
        temp_midpoint = (self.max_temp + self.min_temp) / 2  # midpoint

        # !!placeholder values!!
        prev_indoor_temp = indoor_temp_history[len(indoor_temp_history) - 2]  # get second last temp value
        temp_change = abs(indoor_temp - prev_indoor_temp)

        if self.crit_min_temp <= indoor_temp <= self.crit_max_temp:
            self.crit_time_actual += 1
        else:
            self.crit_time_actual = 0


        # Calculate reward for indoor temperature control
        if self.min_temp <= indoor_temp <= self.max_temp:
            r1 = 1.0
            cooldown_time =+ 1
        elif self.crit_min_temp <= indoor_temp <= self.crit_max_temp and self.crit_time_actual < self.crit_time:
            r1 = 1.0
        elif self.crit_min_temp <= indoor_temp <= self.crit_max_temp and cooldown_time < (self.crit_time*5):
            r1 = (-abs(indoor_temp - temp_midpoint))/2
        else:
            r1 = -abs(indoor_temp - temp_midpoint)

        # Calculate reward for energy savings
        if heating:
            r2 = 0.0
        else:
            r2 = 1.0

        if self.max_allowed_temp_change < (temp_change + 0.1) and self.max_allowed_temp_change < (temp_change - 0.1):
            r3 = 1.0
        else:
            r3 = -abs(self.max_allowed_temp_change - temp_change)

        if cooldown_time >= (self.crit_time*5):
            cooldown_time = 0
        

        # reward weighing
        r1_w = 1.0
        r2_w = 1.0
        r3_w = 1.0

        # Calculate total reward
        r1 *= r1_w
        r2 *= r2_w
        r3 *= r3_w

        total_reward = r1 + r2 + r3

        return total_reward

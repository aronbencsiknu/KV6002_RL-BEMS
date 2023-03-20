

indoor_temp = 20
crit_max_temp = 25
crit_min_temp = 21
crit_time = 5
crit_time_actual = 0


#indoor temp history 


def calculate_reward(self, indoor_temp, indoor_temp_history, energy_consumption, crit_max_temp,crit_min_temp ):

        # reward weighing
        r1_w = 1.0
        r2_w = 1.0
        r3_w = 1.0
        r4_w = 1.0

        #!!placeholder values!!
        max_temp = 20
        min_temp = 15
        temp_midpoint = (max_temp + min_temp) / 2  # midpoint

        # !!placeholder values!!
        max_allowed_temp_change = 5  # degrees/minute
        prev_indoor_temp = indoor_temp_history[len(indoor_temp_history) - 2]  # get second last temp value
        temp_change = abs(indoor_temp - prev_indoor_temp)

        
        if crit_min_temp <= indoor_temp <= crit_max_temp #each call +1 min
            crit_time_actual =+ 1
        else
            crit time_actual = 0
            
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

        if crit_min_temp <= indoor_temp <= crit_max_temp or crit_time_actual > crit_time: #if in critical temp or 
            r4 = -abs((indoor_temp - temp_midpoint)+crit_time_actual) 
        else:
            r4 = 1.0 
            

       
    
    
    # Calculate total reward

        r1 *= r1_w
        r2 *= r2_w
        r3 *= r3_w
        r4 *= r4_w

        total_reward = r1 + r2 + r3 + r4

        #total_reward = -math.log(abs(temp_midpoint - indoor_temp))

        return total_reward
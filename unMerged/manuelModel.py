class manuelControl:
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
    
    def energyConsump(self, last_60_energy):
        rrr = last_60_energy
        file = open("ManualControlEnergy.txt", "a")                          
        file.write(rrr)
        file.close() 





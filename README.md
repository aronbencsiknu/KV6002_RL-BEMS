# Greenhouse Temperature Management with Deep Q Learning

This project was made as part of the KV6002 Team Project and Professionalism module at Northumbria University, Newcastle. 

## Installation instructions

1. [Install Python](https://www.python.org/downloads/)
2. [Install Node.js](https://nodejs.org/en/download)
3. Open the root directory of the project in the command prompt
4. Type ```npm install``` to install the modules required by Node.js
5. Type ```pip install -r requirements.txt``` to install the libraries required by Python

## How to run
After opening the root directory of the project in the command prompt, there are 3 ways to run the project:

### Pre-train
To pre-train a reinforcement learning model, run ```python python_code/main.py -p```.
This trains a model and saves it in the directory trained_model
  
### Local demo
To locally test a model, run ```python python_code/main.py -l```.
This loads a pre-trained model and runs the simulation without slowing down. An example RL model is included in the directory /trained_models. This will also draw a nice matplotlib chart with all the environment observations and agent actions.

### Run GUI
To see the GUI in action, run ```node app.js``` in the root directory.
This starts the server and runs main.py. In this case main.py is called without specifying a flag. Without a specified flag, main.py loads a pre-trained model and runs the simulation. Furthermore, it communicates with the Node.js server through JSON files in directory /public/json. In this case the simulation is slowed down.

## Further explanation

Deep Q-learning is a type of a Reinforcement Learning (RL) algorithm, which is commonly employed to automate and optimize intricate decision-making processes. RL based energy management has been effectively applied in residential and commercial buildings, resulting in a significant decrease in electricity consumption. However, the potential of RL for greenhouse energy management has yet to be fully explored in the scientific literature. My assumption is that the opportunity for energy saving is even more substantial in the case of greenhouses, due to a more flexible, semi time dependent ideal temperature range.
RL algorithms are typically pre-trained in simulated environments, utilizing a numerical reward signal to guide their learning. Subsequently, the pre-trained model can be transferred to real-world scenarios, allowing for quick adaptation to unique environmental features. Notably, RL has the capacity to take into account weather forecasts, representing a critical factor in the significant potential for energy savings.

In this project, the agent is trained on a simulated greenhouse environment with artificially generated weather and weather forecast data. The reward mechanism is based on that outlined in https://www.sciencedirect.com/science/article/pii/S2666546820300434, with some modifications, to account for the intricate heat tolerances of plants. The supervised data for the DQN is generated using the method first proposed in https://arxiv.org/pdf/1312.5602.pdf. Supervised data is obtained using the Bellman optimality equation ...

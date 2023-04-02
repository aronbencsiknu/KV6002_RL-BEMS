# Greenhouse Temperature Management with Deep Q Learning

This project was made as part of the KV6002 Team Project and Professionalism module at Northumbria University, Newcastle. 

## Installation instructions

1. [Install Python 3](https://www.python.org/downloads/)
2. Ensure that the ```python``` terminal command opens python3 by default.
3. [Install Node.js](https://nodejs.org/en/download)
4. Open the root directory of the project in the command prompt
5. Type ```pip install -r requirements.txt``` to install the libraries required by Python

## How to run
After opening the root directory of the project in the command prompt, there are 3 ways to run the project:

### Pre-train
To pre-train a reinforcement learning model, run ```python python_code/main.py -p```.
This trains a model and saves it in the directory trained_model. To enable logging to [wandb.ai](https://wandb.ai/), include the ```-wb``` flag.
  
### Local demo
To locally test a model, run ```python python_code/main.py -l```.
This loads a pre-trained model and runs the simulation without slowing down. An example RL model is included in the directory /trained_models. This will also draw a nice matplotlib chart with all the environment observations and agent actions.

### Run GUI
To see the GUI in action, run ```node app.js``` in the root directory.
This starts the server and runs main.py. In this case main.py is called without specifying a flag. Without a specified flag, main.py loads a pre-trained model and runs the simulation. Furthermore, it communicates with the Node.js server through JSON files in directory /public/json. In this case the simulation is slowed down.

## Database instructions
Firstly, a MySQL server must be installed and configured on the computer. Whilst configuring, use the legacy authentication method and not the default one (The database will not connect otherwise and you will see an error message when running app.js) Set the password to 1234 (If you use any other password, you will need to update that in app.js) After doing so, install MySQL Workbench and launch a database server. Run test.sql in said server. After doing so, change variables enabling (In app.js) and h_logging (in script.js) to true. If you do not turn on and connect your database to the workbench beforehand. The server will send an error 2 Errors you might face: Authentication error when launching the app.js. This means you are not using the legacy version of authentication and reinstallation will be needed for the MySQL server Connection error when launching in app.js. This means you did not launch the server before running the application and therefore cannot connect If causing too many errors, set variables h_logging and useDatabse to False.

## Further explanation

Deep Q-learning is a type of Reinforcement Learning (RL) algorithm, which is commonly employed to automate and optimize intricate decision-making processes. RL-based energy management has been effectively applied in residential and commercial buildings, resulting in a significant decrease in electricity consumption. However, the potential of RL for greenhouse energy management has yet to be fully explored in the scientific literature. I assume that the opportunity for energy saving is even more substantial in the case of greenhouses, due to a more flexible, semi-time-dependent ideal temperature range.
RL algorithms are typically pre-trained in simulated environments, utilizing a numerical reward signal to guide their learning. Subsequently, the pre-trained model can be transferred to real-world scenarios, allowing for quick adaptation to unique environmental features. Notably, RL can take into account weather forecasts, representing a critical factor in the significant potential for energy savings.

# Greenhouse Energy Management with Deep Q Learning

## There are 3 ways to run the project:

1. To pre-train a reinforcement learning model, run "python main.py --pretrain" in directory /python_code.
This trains a model and saves it in the directory trained_model
  
2. To locally test a model, run "python main.py --localdemo" in directory /python_code.
This loads a pre-trained model and runs the simulation without slowing down

3. To see the GUI in action, run "node app.js" in the root directory.
This starts the server and runs main.py. In this case main.py is called without specifying a flag. Without a specified flag, main.py loads a pre-trained model and runs the simulation. Furthermore, it communicates with the Node.js server through JSON files in directory /public/json.

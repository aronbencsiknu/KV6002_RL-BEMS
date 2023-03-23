# Greenhouse Energy Management using Deep Reinforcement Learning

- train.py (Aron) : Training code for the DQN. Contains the experience replay detailed in https://arxiv.org/pdf/1312.5602.pdf.
- environment.py (Jokubas):
- reward.py (Tom):
- GUI (Adam):
- database (Yuelian):
- options.py (Aron) : contains all the hyperparameters.
- plot.py (Jokubas) :
- model.py (Aron) : contains the DQN architecture.
- inference.py (Aron) : loads the saved model. The model is not trained here, no exploring is taking place, only exploiting.

Usage:
1. To train the agent on a simulated greenhouse environment, run train.py.
2. After training saving the model, launch the GUI, which communicates with inference.py



To run The GUI

 You need to install nodeJS before opening the files
open the App.js in Visual Basics Code(Make sure the directory is set to where your file is ),
Type in the terminal node app.js
Click on the link
Gui should be running (Keep in mind when submitting things in desktop2.html, wait a few seconds for the python scripts to run)

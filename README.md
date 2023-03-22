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


# Hyperparameters

NOTIFY_RATE = 5
ACTIONS = 6                 # Number of possible actions to choose from
LEARNING_RATE = 0.00025     # The learning rate
GAMMA = 0.99                # Decay rate of past observations
REPLAY_MEMORY = 25000       # Size of replay memory buffer
BATCH = 128                 # Size of minibatch
EPOCH_SIZE = 50000          # Network Updates per Epoch
EPOCHS = 60                 # Number of Epochs to Run
INITIAL_EPSILON = 0.1       # Initial Epsilon - rate of exploration
FINAL_EPSILONg = 0.05        # Final Episilon
EXPLORE = 3000000           # Timesteps to go from INITIAL_EPSILON to FINAL_EPSILON
OBSERVATION = 1000          # Timesteps to observe before training
ACTION_SPACE_SIZE = 6       # Number of unique actions

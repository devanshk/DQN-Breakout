# Hyperparameters

NOTIFY_RATE = 5
ACTIONS = 6                 # Number of possible actions to choose from
LEARNING_RATE = 0.00025     # The learning rate
GAMMA = 0.99                # Decay rate of past observations
REPLAY_MEMORY = 50000       # Size of replay memory buffer
BATCH = 32                  # Size of minibatch
TRAIN_STEPS = 250000        # Timesteps per Epoch
EPOCHS = 77                 # Number of Epochs to Run
INITIAL_EPSILON = 0.1       # Initial Epsilon - rate of exploration
FINAL_EPSILON = 0.0001      # Final Episilon
EXPLORE = 3000000           # Timesteps to go from INITIAL_EPSILON to FINAL_EPSILON
OBSERVATION = 50000         # Timesteps to observe before training
ACTION_SPACE_SIZE = 6        # Number of unique actions

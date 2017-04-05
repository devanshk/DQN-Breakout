from lib.util import *



def setup():
    """ Initial Setup to use OpenAI's Breakout environment """
    global env
    env = gym.make('Breakout-v0')

    print("action space: ", env.action_space)
    print("observation space: ", env.observation_space)

def pre_process(observ, init=False):
    """ Pre-process the image and prep it for the network """
    global s_t, s_t1

    x_t = skimage.color.rgb2gray(observ) # Grayscale
    x_t = skimage.transform.resize(x_t, (110,84), mode='constant', preserve_range=True) # Downsample
    x_t = skimage.util.crop(x_t,((19,7),(0,0))) # Crop
    # x_t = skimage.exposure.rescale_intensity(x_t, out_range=(0, 255)) # Threshold as black and white

    if (init): # If it's the beginning of our episode
        # Create the 4-screen state stack
        s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)
        s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])

    else: # Otherwise add it to our state stack
        x_t1 = x_t.reshape(1, x_t.shape[0], x_t.shape[1], 1)

        s_t1 = np.append(x_t1, s_t[:, :, :, :3], axis=3)

def buildmodel():
    """ Build the actual model """
    print("Building Model...")
    model = Sequential()
    model.add(Conv2D(32, (8, 8), strides=(4,4), input_shape=(84,84,4), padding='same', name='conv1'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (4, 4), strides=(2,2), padding='same', name='conv2'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), strides=(1,1), padding='same', name='conv3'))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512, name='fc1'))
    model.add(Activation('relu'))
    model.add(Dense(ACTION_SPACE_SIZE, name='fc2'))

    adam = Adam(lr = LEARNING_RATE)
    model.compile(loss='mse', optimizer = adam)
    return model

def setupEpisode():
    """ Do some initial setup for the first episode """
    global epsilon, env
    env.reset() # Take an initial step
    observation, reward, done, info = env.step(5)
    pre_process(observation, True)
    epsilon = INITIAL_EPSILON


def trainNet(model, args):
    """ Train the network to play Breakout """
    global epsilon, observation, reward, done, info, env, s_t, s_t1

    ep = 1

    while(ep <= EPOCHS):
        start = time.time() # Time each epoch
        qMax = 0 # Store Best Q info
        t = 0
        D = deque()
        setupEpisode()
        while(t < EPOCH_SIZE):
            loss = 0        # Loss
            Q_sa = 0        # Q value given state/action pair
            action = 0      # Action
            r_t = 0         # Reward

            # Limit Rendering ot the first 500 timesteps of each epoch
            if (args['render']):
                env.render()

            # Every so often, based on exploration rate, take a random action
            if random.random() <= epsilon:
                action = env.action_space.sample()
            else: # If we're not taking a random action, let the network decide
                q = model.predict(s_t)
                max_Q = np.argmax(q)
                action = max_Q

            # Gradually reduce the Exploration probability
            if epsilon > FINAL_EPSILON and t > OBSERVATION:
                epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

            # Take the action and observe the result
            observation, reward, done, info = env.step(action)

            # Pre-Process the new state
            pre_process(observation)

            # Store the transition in Replay Memory
            D.append((s_t, action, reward, s_t1, done))
            if len(D) > REPLAY_MEMORY: # And keep to our Replay Memory Size constraints
                D.popleft()

            if (done):
                setupEpisode()
            # print("start train")
            # Train after observing for a period of timesteps
            if t == OBSERVATION:
                print("--Observation Over, Training Now--")
            if t > OBSERVATION:
                # Sample a minibatch to train on
                minibatch = random.sample(D, BATCH)

                inputs = np.zeros((BATCH, s_t.shape[1], s_t.shape[2], s_t.shape[3]))
                targets = np.zeros((inputs.shape[0], ACTIONS))

                # Use Experience Replay
                for i in range(0, len(minibatch)):
                    state_t = minibatch[i][0]
                    action_t = minibatch[i][1]
                    reward_t = minibatch[i][2]
                    state_t1 = minibatch[i][3]
                    done = minibatch[i][4]

                    inputs[i : i+1] = state_t # Save s_t

                    targets[i] = model.predict(state_t) # Q-values for each action
                    Q_sa = model.predict(state_t1)

                    if done: # If the game's over, the reward is just the current reward
                        targets[i, action_t] = reward_t
                    else: # Otherwise we use our approximated Q function
                        targets[i, action_t] = reward_t + GAMMA * np.max(Q_sa)

                loss += model.train_on_batch(inputs, targets)
            # print("end train")

            # Update the state and time
            s_t = s_t1
            t = t+1

            # Save our model every 1000 iterations
            if t % 1000 == 0 and args['save']:
                # print("Saved model.")
                model.save_weights("model.h5", overwrite = True)
                with open("model.json","w") as outfile:
                    json.dump(model.to_json(), outfile)

            if t%10000 == 0:
                print ("Time: ",t//10000)

            # Debug Info
            state = ""
            if t <= OBSERVATION:
                state = "observe"
            elif t > OBSERVATION and t <= OBSERVATION+EXPLORE:
                state = "explore"
            else:
                state = "train"

            bestQ = np.max(Q_sa)
            if (bestQ > qMax):
                qMax = bestQ

            # print("TIMESTEP", t, "/ STATE", state, \
            #     "/ EPSILON", epsilon, "/ ACTION", action, "/ REWARD", r_t, \
            #     "/ Q_MAX " , np.max(Q_sa), "/ Loss ", loss)


        print("Epoch {} finished in {}. Q_MAX: {}".format(ep, time.time()-start, qMax))
        print("*****************")
        ep = ep + 1


def main():
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--load_weights', help='Specify a model to load', required=False)
    parser.add_argument("--render", help="Render the game", action="store_true")
    parser.add_argument("--save", help="Save the model", action="store_true")
    args = vars(parser.parse_args())

    net = buildmodel()

    if (args['load_weights']):
        print("Loading model from ",args['load_weights'])
        net.load_weights(args['load_weights'])
    else:
        print("Using Fresh Model.")

    if (args['render']):
        print("Render enabled.")
    if (args['save']):
        print("Save enabled.")

    setup()
    trainNet(net, args)

if __name__ == "__main__":
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    from keras import backend as K
    K.set_session(sess)
    K.set_image_dim_ordering('tf')
    main()

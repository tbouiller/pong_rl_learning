import os
import numpy as np
import random
import warnings
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML, clear_output
import matplotlib
from ple import PLE
from ple.games.pong import Pong

matplotlib.rcParams['animation.embed_limit'] = 2**128
warnings.filterwarnings('ignore')
os.environ[
    "CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # No cuda available on personal laptop
os.environ["CUDA_VISIBLE_DEVICES"] = ""

os.putenv('SDL_VIDEODRIVER', 'fbcon')  # settings for pygame display
os.environ["SDL_VIDEODRIVER"] = "dummy"

NORMALIZE_FACTORS = np.array([48, 50, 48, 64, 48, 50, 50])


def process_state(state):
    state = np.array(list(state.values()))
    state /= NORMALIZE_FACTORS

    return state


# Setting up the game environment, refer to the PLE docs if you want
# to know the details
game = Pong(MAX_SCORE=7)
game_env = PLE(game,
               fps=30,
               display_screen=False,
               state_preprocessor=process_state,
               reward_values={
                   "win": 0,
                   "loss": 0
               })


def render(episode):

    fig = plt.figure()
    img = plt.imshow(np.transpose(episode[0], [1, 0, 2]))
    plt.axis('off')

    def animate(i):
        img.set_data(np.transpose(episode[i], [1, 0, 2]))
        return img,

    anim = FuncAnimation(fig,
                         animate,
                         frames=len(episode),
                         interval=24,
                         blit=True)
    html = HTML(anim.to_jshtml())

    return html


class Results(dict):
    def __init__(self, *args, **kwargs):
        if 'filename' in kwargs:
            data = np.load(kwargs['filename'])
            super(Results, self).__init__(data)
        else:
            super(Results, self).__init__(*args, **kwargs)
        self.new_key = None
        self.plot_keys = None
        self.ylim = None

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.new_key = key

    def plot(self, window):

        clear_output(wait=True)
        for key in self:
            # Ensure latest results are plotted on top
            if self.plot_keys is not None and key not in self.plot_keys:
                continue
            elif key == self.new_key:
                continue
            self.plot_smooth(key, window)
        if self.new_key is not None:
            self.plot_smooth(self.new_key, window)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.legend(loc='lower right')
        if self.ylim is not None:
            plt.ylim(self.ylim)
        plt.show()

    def plot_smooth(self, key, window):
        if len(self[key]) == 0:
            plt.plot([], [], label=key)
            return None
        y = np.convolve(self[key], np.ones((window, )) / window, mode='valid')
        x = np.linspace(window / 2, len(self[key]) - window / 2, len(y))
        plt.plot(x, y, label=key)

    def save(self, filename='results'):
        np.savez(filename, **self)


def run_fixed_episode(env, policy):
    frames = []
    env.reset_game()
    done = False
    while not done:
        observation = env.getGameState()
        action = policy(env, observation)
        frames.append(env.getScreenRGB())
        reward = env.act(action)
        done = env.game_over()
    return frames


def random_policy(env, observation):
    return random.sample(env.getActionSet(), 1)[0]


def run_fixed_episode_learned(env, policy):
    frames = []
    env.reset_game()
    done = False
    while not done:
        observation = env.getGameState()
        action_idx = policy.decide(observation)
        action = env.getActionSet()[action_idx]
        frames.append(env.getScreenRGB())
        # reward = env.act(action)
        done = env.game_over()
    return frames


num_episodes = 5000


def run_experiment(experiment_name,
                   env,
                   num_episodes,
                   reward_shaping=False,
                   policy_learning_rate=0.001,
                   value_learning_rate=0.001,
                   baseline=None,
                   fileNamePolicy=None,
                   fileNameValue=None,
                   verbose=False,
                   stopping_criterion=20):

    env.init()
    discount_factor = 0.99

    #Initiate the learning agent
    agent = RLAgent(n_obs=env.getGameStateDims()[0],
                    policy_learning_rate=policy_learning_rate,
                    value_learning_rate=value_learning_rate,
                    discount=discount_factor,
                    baseline=baseline,
                    fileNamePolicy=fileNamePolicy,
                    fileNameValue=fileNameValue)

    rewards = []
    all_episode_frames = []

    points_won = 0
    games_won = 0
    win_streak = 0

    for episode in range(1, num_episodes + 1):

        #Update results plot and occasionally store an episode movie
        episode_frames = None

        if episode % 10 == 0:
            results[experiment_name] = np.array(rewards)
            #results.plot(10)
            if verbose:
                print("Number of games won: " + str(int(games_won)))
                print("Number of points won: " + str(int(points_won)))

        if episode % 500 == 0:
            episode_frames = []

        #Reset the environment for a new episode
        env.reset_game()

        observation = env.getGameState()

        player_points = 0
        opponent_points = 0

        episode_steps = 0
        episode_reward = 0

        while True:

            if episode_frames is not None:
                episode_frames.append(env.getScreenRGB())

            # 1. Decide on an action based on the observations
            action_idx = agent.decide(observation)
            # convert action index into commands expected by the game environment
            action = game_env.getActionSet()[action_idx]

            # 2. Take action in the environment
            raw_reward = env.act(action)
            next_observation = env.getGameState()

            if raw_reward > 0:
                points_won += raw_reward
                player_points += raw_reward
            elif raw_reward < 0:
                opponent_points += np.abs(raw_reward)

            episode_steps += 1

            # 3. Reward shaping
            if reward_shaping:
                auxiliary_reward = reward_design(observation)
                reward = raw_reward + auxiliary_reward
            else:
                reward = raw_reward

            episode_reward += reward

            # 4. Store the information returned from the environment for training
            agent.observe(observation, action_idx, reward)

            # 5. When we reach a terminal state ("done"), use the observed episode to train the network
            done = env.game_over()  # Check if game is over
            # print(agent.episode_observations[-1])
            if done:
                rewards.append(episode_reward)
                agent.train()

                # Some diagnostics
                if verbose:
                    print("Game score: " + str(int(player_points)) + "-" +
                          str(int(opponent_points)) + " over " +
                          str(episode_steps) + " frames")

                # Calculating the win streak (number of consecutive games won)
                if player_points > opponent_points:
                    print("Won a game at episode " + str(episode) + "!")
                    games_won += 1
                    win_streak += 1
                else:
                    win_streak = 0

                if episode_frames is not None:
                    all_episode_frames.append(episode_frames)

                break

            # Reset for next step
            observation = next_observation

        # Stop if you won enough consecutive games
        if win_streak == stopping_criterion:
            break
    results.plot(10)
    return all_episode_frames, agent


def reward_design(observation):
    ball_vel = observation[5]

    auxiliary_reward = 0
    if ball_vel > 0:
        auxiliary_reward = 1e-3
    return auxiliary_reward


class RLAgent(object):
    def __init__(self,
                 n_obs,
                 policy_learning_rate,
                 value_learning_rate,
                 discount,
                 baseline=None,
                 fileNamePolicy=None,
                 fileNameValue=None):

        #We need the state and action dimensions to build the network
        self.n_obs = n_obs
        self.n_act = 1

        self.gamma = discount
        self.use_baseline = None
        if baseline is not None:
            self.use_baseline = baseline

        self.use_adaptive_baseline = baseline == 'adaptive'

        #Fill in the rest of the agent parameters to use in the methods below

        # TODO

        #These lists stores the observations for this episode
        self.episode_observations, self.episode_actions, self.episode_rewards = [], [], []

        #Build the keras network
        self.fileNamePolicy = fileNamePolicy
        self.fileNameValue = fileNameValue
        self._build_network()

    def observe(self, state, action, reward):
        """ This function takes the observations the agent received from the environment and stores them
            in the lists above. """
        self.episode_observations.append(state)
        actions = np.zeros(2)
        actions[action] = 1
        self.episode_actions.append(actions)
        self.episode_rewards.append(reward)

    def _get_returns(self):
        """ This function should process self.episode_rewards and return the discounted episode returns
            at each step in the episode, then optionally apply a baseline. Hint: work backwards."""

        returns = np.zeros_like(self.episode_rewards)
        running_add = 0

        for t in reversed(range(len(returns))):
            running_add = running_add * self.gamma + self.episode_rewards[t]
            returns[t] = running_add

        if self.use_baseline == 'simple':
            returns -= [np.mean(returns[i:]) for i in range(len(returns))]

        #Normalization
        returns = (returns - np.mean(returns)) / np.std(returns)

        if self.use_adaptive_baseline == True:
            value_return = []
            for obs in self.episode_observations:
                value_return.append([
                    None,
                    np.squeeze(self.value_net.predict(np.array([obs])))
                ])

            return returns, np.array(value_return)

        return returns

    def _build_network(self):
        """ This function should build the network that can then be called by decide and train. 
            The network takes observations as inputs and has a policy distribution as output."""
        self.policy_net = Sequential()
        # Hidden Layers
        self.policy_net.add(Dense(32, input_shape=(7, ), activation='relu'))
        self.policy_net.add(Dense(32, activation='relu'))
        self.policy_net.add(Dense(32, activation='relu'))
        # Output Layer
        self.policy_net.add(Dense(2, activation='softmax'))
        self.policy_net.compile(loss='binary_crossentropy',
                                optimizer=Adam(lr=policy_learning_rate),
                                metrics=['accuracy'])

        if self.use_adaptive_baseline == True:
            self.value_net = Sequential()

            self.value_net.add(Dense(32, input_shape=(7, ), activation='relu'))
            self.value_net.add(Dense(32, activation='relu'))
            self.value_net.add(Dense(32, activation='relu'))

            self.value_net.add(Dense(1))

            self.value_net.compile(loss='mean_squared_error',
                                   optimizer=Adam(lr=value_learning_rate),
                                   metrics=['accuracy'])

    def decide(self, state):
        """ This function feeds the observed state to the network, which returns a distribution
            over possible actions. Sample an action from the distribution and return it."""
        if state.shape != (7, ):
            raise TypeError("Wrong state shape is given: {}".format(state))
        action_prob = np.squeeze(self.policy_net.predict(np.array([state])))

        assert len(action_prob) == 2, "{} != {}".format(len(action_prob), 2)
        #Sampling and return action
        return np.random.choice(np.arange(2), p=action_prob)
        #return np.argmax(self.policy_net.predict(state))

    def train(self):
        """ When this function is called, the accumulated observations, actions and discounted rewards from the
            current episode should be fed into the network and used for training. Use the _get_returns function 
            to first turn the episode rewards into discounted returns. """
        discounted_returns, value_r = self._get_returns()

        #Encourage or not the action based on the returns that we get
        self.policy_net.train_on_batch(np.array(self.episode_observations),
                                       np.array(self.episode_actions),
                                       sample_weight=(discounted_returns -
                                                      value_r[:, 1]))

        if self.use_adaptive_baseline == True:
            self.value_net.train_on_batch(np.array(self.episode_observations),
                                          discounted_returns)
        self.episode_observations = []
        self.episode_actions = []
        self.episode_rewards = []


policy_learning_rate = 1e-3
value_learning_rate = 1e-3
results = Results()
_, adaptive_policy = run_experiment("REINFORCE (adaptive baseline)",
                                    game_env,
                                    num_episodes,
                                    reward_shaping=False,
                                    policy_learning_rate=policy_learning_rate,
                                    value_learning_rate=value_learning_rate,
                                    stopping_criterion=30,
                                    baseline='adaptive',
                                    verbose=True)

results.save('results_exercise_2')
adaptive_policy.policy_net.save('policy_net_disc.h5')
adaptive_policy.value_net.save('value_net_disc.h5')


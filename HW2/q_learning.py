import numpy as np
import operator
#import matplotlib.pyplot as plt

flag1=0
flag2=0
class GridWorld:
    def __init__(self):
        # Set information about the gridworld
        self.rows = 5
        self.columns = 10
        self.grid = np.zeros((self.rows, self.columns)) - 1

        # Set random start location for the agent
        self.agent_location = (2,3)
        #self.agent2_location= (2,3)
        #self.agent_location=(2,3)
        self.flag1=0
        self.flag2=0

        
        self.target1_location = (3, 1)
        self.target2_location = (0,9)
        #self.target_states = [self.target1_location]
        self.target_states = [self.target1_location,self.target2_location]
        # Set grid rewards for special cells

        self.grid[self.target1_location[0], self.target1_location[1]] = 20
        self.grid[self.target2_location[0], self.target2_location[1]]= 20

        # Set available actions
        self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']

    def get_available_actions(self):
        """Returns possible actions"""

        return self.actions

    def agent_on_map(self):
        """Prints out current location of the agent on the grid (used for debugging)"""
        grid = np.zeros((self.rows, self.columns))
        grid[self.agent_location[0], self.agent_location[1]] = 1
        #grid[self.agent2_location[0], self.agent2_location[1]] = 2
        return grid
    def get_reward(self, new_location):
        """Returns the reward for an input position"""
        return self.grid[new_location[0], new_location[1]]

    def make_step(self, action):
        """Moves the agent in the specified direction. If agent is at a border, agent stays still
        but takes negative reward. Function returns the reward for the move."""
        # Store previous location
        last_location = self.agent_location

        if action == 'UP':
            # If agent is at the top, stay still, collect reward
            if last_location[0] == 0:
                #Q_Agent.choose_action()
                reward = self.get_reward(last_location)
            else:
                self.agent_location = (self.agent_location[0] - 1, self.agent_location[1])
                #self.agent_location = (self.agent_location[0] - 1, self.agent_location[1])
                #self.agent2_location = (self.agent2_location[0] - 1, self.agent2_location[1])
                reward = self.get_reward(self.agent_location)

            # DOWN
        elif action == 'DOWN':
            # If agent is at bottom, stay still, collect reward
            if last_location[0] == self.rows - 1:
                #Q_Agent.choose_action()
                reward = self.get_reward(last_location)
            else:
                self.agent_location = (self.agent_location[0] + 1, self.agent_location[1])
                reward = self.get_reward(self.agent_location)

            # LEFT
        elif action == 'LEFT':
            # If agent is at the left, stay still, collect reward
            if last_location[1] == 0:
                #Q_Agent.choose_action()
                reward = self.get_reward(last_location)
            else:
                self.agent_location = (self.agent_location[0], self.agent_location[1] - 1)
                reward = self.get_reward(self.agent_location)

            # RIGHT
        elif action == 'RIGHT':
            # If agent is at the right, stay still, collect reward
            if last_location[1] == self.columns - 1:
                #Q_Agent.choose_action()
                reward = self.get_reward(last_location)
            else:
                self.agent_location = (self.agent_location[0], self.agent_location[1] + 1)
                reward = self.get_reward(self.agent_location)

        return reward

    def check_state(self):

        """Check if the agent is in a terminal state, if so return 'TERMINAL'"""
        #if self.agent_location in self.target_states[0] :
            #self.flag1+=1
        #if self.agent_location in self.target_states[1]:
            #self.flag2+=1
        #if self.flag1 ==1:
        if self.agent_location in self.target_states:
            return 'END'
class Q_Agent():
    # Intialise
    def __init__(self, environment, epsilon=0.05, alpha=0.1, gamma=1):
        self.environment = environment
        self.q_table = dict() # Store all Q-values in dictionary of dictionaries
        for x in range(environment.rows): # Loop through all possible grid spaces, create sub-dictionary for each
            for y in range(environment.columns):
                self.q_table[(x,y)] = {'UP':0, 'DOWN':0, 'LEFT':0, 'RIGHT':0} # Populate sub-dictionary with zero values for possible moves

        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.num_agents= 2

    def choose_action(self, available_actions):
        """Returns the optimal action from Q-Value table. If multiple optimal actions, chooses random choice.
        Will make an exploratory random action dependent on epsilon."""
        if np.random.uniform(0, 1) < self.epsilon:
            action = available_actions[np.random.randint(0, len(available_actions))]
            if action == 'UP' and GridWorld().agent_location[0]==0:
                action = available_actions[np.random.randint(0, len(available_actions))]
            if action == 'DOWN' and GridWorld().agent_location[0]==GridWorld().rows-1:
                action = available_actions[np.random.randint(0, len(available_actions))]
            if action == 'LEFT' and GridWorld().agent_location[1]==0:
                action = available_actions[np.random.randint(0, len(available_actions))]
            if action == 'RIGHT' and GridWorld().agent_location[1]==GridWorld().columns-1:
                action = available_actions[np.random.randint(0, len(available_actions))]


        else:
            q_values_of_state = self.q_table[self.environment.agent_location]
            maxValue = max(q_values_of_state.values())
            action = np.random.choice([k for k, v in q_values_of_state.items() if v == maxValue])


        return action

    def learn(self, old_state, reward, new_state, action):
        """Updates the Q-value table using Q-learning"""
        q_values_of_state = self.q_table[new_state]
        max_q_value_in_new_state = max(q_values_of_state.values())
        current_q_value = self.q_table[old_state][action]

        self.q_table[old_state][action] = (1 - self.alpha) * current_q_value + self.alpha * (
                    reward + self.gamma * max_q_value_in_new_state)

    def play(self, agent, trials):
        """The play function runs iterations and updates Q-values if desired."""
        reward_per_episode = []  # Initialise performance log
        flag1=0
        flag2=0

        for trial in range(trials):  # Run trials
            cumulative_reward = 0  # Initialise values of each game
            step = 0
            game_over = False
            while game_over != True:  # Run until max steps or until game is finished
                old_state = self.environment.agent_location
                action = agent.choose_action(self.environment.actions)
                #action= agent2.choose_action(self.environment.actions)
                reward = self.environment.make_step(action)
                new_state = self.environment.agent_location


                agent.learn(old_state, reward, new_state, action)
                #agent2.learn(old_state, reward, new_state, action)

                cumulative_reward += reward
                step += 1
                #print("Current position of the agent =", self.environment.agent_location)
                #print(self.environment.agent_on_map())
                #print(reward)


                if self.environment.check_state() == 'END':  # If game is in terminal state, game over and start next trial
                    print("Current position of the agent =", self.environment.agent_location)
                    print(self.environment.agent_on_map())
                    print("number of steps=",step)
                    print("reward=",reward)
                    self.environment.__init__()
                    game_over = True

            reward_per_episode.append(cumulative_reward)  # Append reward for current trial to performance log

        return reward_per_episode  # Return performance log



environment = GridWorld()
agentQ1 = Q_Agent(environment)
agentQ2 = Q_Agent(environment)

reward_per_episode = agentQ1.play(agentQ1, trials=20)
#plt.plot(reward_per_episode)
#reward_per_episode= agentQ2.play(agentQ2, trials=2)

"""
    print("Current position of the agent =", environment.agent_location)
    print(environment.agent_on_map())
    available_actions = environment.get_available_actions()
    print("Available_actions =", available_actions)
    chosen_action = agentQ.choose_action(available_actions)
    print("Randomly chosen action =", chosen_action)
    reward = environment.make_step(chosen_action)
    print("Reward obtained =", reward)
    print("Current position of the agent =", environment.agent_location)
    print(environment.agent_on_map())
"""

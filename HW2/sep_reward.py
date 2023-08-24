import numpy as np
import operator
class GridWorld:
    def __init__(self):
        # Set information about the gridworld
        self.rows = 5
        self.columns = 10
        self.grid = np.zeros((self.rows, self.columns)) - 1

        # Set random start location for the agent
        self.agent1_location = (2,3)
        self.agent2_location= (2,3)
        #self.agent_location=(2,3)
        self.flag1=0
        self.flag2=0

        
        self.target1_location = (3, 1)
        self.target2_location = (0,9)
        self.target_states = [self.target1_location,self.target2_location]
        # Set grid rewards for special cells

        self.grid[self.target1_location[0], self.target1_location[1]] = 20
        self.grid[self.target2_location[0], self.target2_location[1]]= 20

        # Set available actions
        self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']

    def agent1_get_available_actions(self):
        """Returns possible actions"""

        return self.actions

    def agent1_on_map(self):
        """Prints out current location of the agent on the grid (used for debugging)"""
        grid = np.zeros((self.rows, self.columns))
        grid[self.agent1_location[0], self.agent1_location[1]] = 1
        #grid[self.agent2_location[0], self.agent2_location[1]] = 2
        return grid
    def agent2_on_map(self):
        """Prints out current location of the agent on the grid (used for debugging)"""
        grid = np.zeros((self.rows, self.columns))
        grid[self.agent2_location[0], self.agent2_location[1]] = 2
        #grid[self.agent2_location[0], self.agent2_location[1]] = 2
        return grid
    def agent1_get_reward(self, agent1_location):
        """Returns the reward for an input position"""
        return self.grid[agent1_location[0], agent1_location[1]]
    def agent2_get_reward(self, agent2_location):
        """Returns the reward for an input position"""
        return self.grid[agent2_location[0], agent2_location[1]]

    def agent1_make_step(self, action):
        """Moves the agent in the specified direction. If agent is at a border, agent stays still
        but takes negative reward. Function returns the reward for the move."""
        # Store previous location
        last_location_1 = self.agent1_location
        last_location_2= self.agent2_location

        if action == 'UP':
            # If agent is at the top, stay still, collect reward
            if last_location_1[0] == 0:
                #Q_Agent.choose_action()
                reward = self.agent1_get_reward(last_location_1)
            else:
                self.agent1_location = (self.agent1_location[0] - 1, self.agent1_location[1])
                #self.agent_location = (self.agent_location[0] - 1, self.agent_location[1])
                #self.agent2_location = (self.agent2_location[0] - 1, self.agent2_location[1])
                reward = self.agent1_get_reward(self.agent1_location)

            # DOWN
        elif action == 'DOWN':
            # If agent is at bottom, stay still, collect reward
            if last_location_1[0] == self.rows - 1:
                #Q_Agent.choose_action()
                reward = self.agent1_get_reward(last_location_1)
            else:
                self.agent1_location = (self.agent1_location[0] + 1, self.agent1_location[1])
                reward = self.agent1_get_reward(self.agent1_location)

            # LEFT
        elif action == 'LEFT':
            # If agent is at the left, stay still, collect reward
            if last_location_1[1] == 0:
                #Q_Agent.choose_action()
                reward = self.agent1_get_reward(last_location_1)
            else:
                self.agent1_location = (self.agent1_location[0], self.agent1_location[1] - 1)
                reward = self.agent1_get_reward(self.agent1_location)

            # RIGHT
        elif action == 'RIGHT':
            # If agent is at the right, stay still, collect reward
            if last_location_1[1] == self.columns - 1:
                #Q_Agent.choose_action()
                reward = self.agent1_get_reward(last_location_1)
            else:
                self.agent1_location = (self.agent1_location[0], self.agent1_location[1] + 1)
                reward = self.agent1_get_reward(self.agent1_location)

        return reward
    def agent2_make_step(self, action):
        """Moves the agent in the specified direction. If agent is at a border, agent stays still
        but takes negative reward. Function returns the reward for the move."""
        # Store previous location
        last_location_1 = self.agent1_location
        last_location_2 = self.agent2_location

        if action == 'UP':
            # If agent is at the top, stay still, collect reward
            if last_location_2[0] == 0:
                #Q_Agent.choose_action()
                reward = self.agent2_get_reward(last_location_2)
            else:
                self.agent2_location = (self.agent2_location[0] - 1, self.agent2_location[1])
                #self.agent2_location = (self.agent2_location[0] - 1, self.agent_location[1])
                #self.agent2_location = (self.agent2_location[0] - 1, self.agent2_location[1])
                reward = self.agent2_get_reward(self.agent2_location)

            # DOWN
        elif action == 'DOWN':
            # If agent is at bottom, stay still, collect reward
            if last_location_2[0] == self.rows - 1:
                #Q_Agent.choose_action()
                reward = self.agent2_get_reward(last_location_2)
            else:
                self.agent2_location = (self.agent2_location[0] + 1, self.agent2_location[1])
                reward = self.agent2_get_reward(self.agent2_location)

            # LEFT
        elif action == 'LEFT':
            # If agent is at the left, stay still, collect reward
            if last_location_2[1] == 0:
                #Q_Agent.choose_action()
                reward = self.agent2_get_reward(last_location_2)
            else:
                self.agent2_location = (self.agent2_location[0], self.agent1_location[1] - 1)
                reward = self.agent2_get_reward(self.agent2_location)

            # RIGHT
        elif action == 'RIGHT':
            # If agent is at the right, stay still, collect reward
            if last_location_2[1] == self.columns - 1:
                #Q_Agent.choose_action()
                reward = self.agent2_get_reward(last_location_2)
            else:
                self.agent2_location = (self.agent2_location[0], self.agent2_location[1] + 1)
                reward = self.agent2_get_reward(self.agent2_location)

        return reward
    def agent1_check_state(self):

        """Check if the agent is in a terminal state, if so return 'TERMINAL'"""
        #if self.agent_location in self.target_states[0] :
            #self.flag1+=1
        #if self.agent_location in self.target_states[1]:
            #self.flag2+=1
        #if self.flag1 ==1:
        if self.agent1_location in self.target_states:
        #and self.agent2_location in self.target_states:
            return 'END'
    def agent2_check_state(self):

        """Check if the agent is in a terminal state, if so return 'TERMINAL'"""
        #if self.agent_location in self.target_states[0] :
            #self.flag1+=1
        #if self.agent_location in self.target_states[1]:
            #self.flag2+=1
        #if self.flag1 ==1:
        if self.agent2_location in self.target_states:
        #and self.agent2_location in self.target_states:
            return 'END'
class Q_Agent_1():
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
            if action == 'UP' and GridWorld().agent1_location[0]==0:
                action = available_actions[np.random.randint(0, len(available_actions))]
            if action == 'DOWN' and GridWorld().agent1_location[0]==GridWorld().rows-1:
                action = available_actions[np.random.randint(0, len(available_actions))]
            if action == 'LEFT' and GridWorld().agent1_location[1]==0:
                action = available_actions[np.random.randint(0, len(available_actions))]
            if action == 'RIGHT' and GridWorld().agent1_location[1]==GridWorld().columns-1:
                action = available_actions[np.random.randint(0, len(available_actions))]


        else:
            q_values_of_state = self.q_table[self.environment.agent1_location]
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
                old_state = self.environment.agent1_location
                action = agent.choose_action(self.environment.actions)
                #action= agent2.choose_action(self.environment.actions)
                reward = self.environment.agent1_make_step(action)
                new_state = self.environment.agent1_location


                agent.learn(old_state, reward, new_state, action)
                #agent2.learn(old_state, reward, new_state, action)

                cumulative_reward += reward
                step += 1
                #print("Current position of the agent =", self.environment.agent1_location)
                #print(self.environment.agent1_on_map())
                #print(reward)


                if self.environment.agent1_check_state() == 'END':  # If game is in terminal state, game over and start next trial
                    print("Current position of the agent =", self.environment.agent1_location)
                    print(self.environment.agent1_on_map())
                    print("number of steps=",step)
                    print("reward=",reward)
                    self.environment.__init__()
                    game_over = True

            reward_per_episode.append(cumulative_reward)  # Append reward for current trial to performance log

        return reward_per_episode  # Return performance log
class Q_Agent_2():
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
            if action == 'UP' and GridWorld().agent2_location[0]==0:
                action = available_actions[np.random.randint(0, len(available_actions))]
            if action == 'DOWN' and GridWorld().agent2_location[0]==GridWorld().rows-1:
                action = available_actions[np.random.randint(0, len(available_actions))]
            if action == 'LEFT' and GridWorld().agent2_location[1]==0:
                action = available_actions[np.random.randint(0, len(available_actions))]
            if action == 'RIGHT' and GridWorld().agent2_location[1]==GridWorld().columns-1:
                action = available_actions[np.random.randint(0, len(available_actions))]


        else:
            q_values_of_state = self.q_table[self.environment.agent2_location]
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
class Play_game:
    def __init__(self, environment,agent1,agent2):
        self.environment=environment
        self.agent1= agent1
        self.agent2= agent2

    def play(self,environment,agent1,agent2, trials,max_steps_per_episode=10000):
        """The play function runs iterations and updates Q-values if desired."""
        reward_per_episode = []  # Initialise performance log
        flag1=0
        flag2=0

        for trial in range(trials):  # Run trials
            cumulative_reward = 0  # Initialise values of each game
            step = 0
            game_over = False
            while step < max_steps_per_episode and game_over != True:  # Run until max steps or until game is finished
                old_state_1 = self.environment.agent1_location
                old_state_2 = self.environment.agent2_location
                action_1 = self.agent1.choose_action(self.environment.actions)
                action_2= self.agent2.choose_action(self.environment.actions)
                reward_1 = self.environment.agent1_make_step(action_1)
                reward_2= self.environment.agent2_make_step(action_2)
                new_state_1 = self.environment.agent1_location
                new_state_2= self.environment.agent2_location


                self.agent1.learn(old_state_1, reward_1, new_state_1, action_1)
                self.agent2.learn(old_state_2, reward_2, new_state_2, action_2)

                cumulative_reward += reward_1+ reward_2
                step += 1
                print("Current position of the agent1 =", self.environment.agent2_location)
                print("Current position of the agent2 =", self.environment.agent1_location)
                print(self.environment.agent2_on_map())
                print(self.environment.agent1_on_map())
                print(reward_1)
                print(reward_2)
                print(step)


                if self.environment.agent1_check_state() == 'END' or self.environment.agent2_check_state() == 'END':  # If game is in terminal state, game over and start next trial
                    print("Current position of the agent =", self.environment.agent2_location)
                    print(self.environment.agent1_on_map())
                    print(self.environment.agent2_on_map())
                    print("Current position of the agent1 =", self.environment.agent2_location)
                    print("Current position of the agent2 =", self.environment.agent1_location)
                    print("number of steps=",step)
                    print("reward1=",reward_1)
                    print("reward2=",reward_2)
                    self.environment.__init__()
                    game_over = True

            reward_per_episode.append(cumulative_reward)  # Append reward for current trial to performance log

        return reward_per_episode  # Return performance log
environment = GridWorld()
agentQ1 = Q_Agent_1(environment)
agentQ2 = Q_Agent_2(environment)
game= Play_game(environment,agentQ1,agentQ2)
reward_per_episode = game.play(environment,agentQ1,agentQ2, trials=1, max_steps_per_episode=10000)


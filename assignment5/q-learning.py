import gym
import gym_toytext
import numpy as np
import random
env = gym.make("NChain-v0")
env.action_space.sample()

num_episodes = 3000 #20000 #60000
gamma = 0.95 #0.99
learning_rate = 0.2 #0.95 #0.85
epsilon = 0.5#1 #0.15 #0.1

# initialize the Q table
# Q = np.zeros([5,2])

# for _ in range(num_episodes):
# 	state = env.reset()
# 	done = False
# 	while done == False:
#         # First we select an action:
# 		if random.uniform(0, 1) < epsilon: # Flip a skewed coin
# 			action = env.action_space.sample() # Explore action space
# 		else:
# 			action = np.argmax(Q[state,:]) # Exploit learned values
#         # Then we perform the action and receive the feedback from the environment
# 		new_state, reward, done, info = env.step(action)
#         # Finally we learn from the experience by updating the Q-value of the selected action
# 		update = reward + (gamma*np.max(Q[new_state,:])) - Q[state, action]
# 		Q[state,action] += learning_rate*update 
# 		state = new_state
# np.save("Q.npy",Q)
Q = np.load("Q.npy")
# state = env.reset()
# for step in range(10):
#     env.render()
#     # Take the action (index) with the maximum expected discounted future reward given that state
#     action = np.argmax(Q[state,:])
#     state, reward, done, info = env.step(action)
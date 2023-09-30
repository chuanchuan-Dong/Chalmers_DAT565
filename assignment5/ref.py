import numpy as np

def value_iteration(epsilon, rewards, transition_probabilities, gamma):
    num_states = len(rewards)
    V = np.zeros(num_states)  # Initialize the value function with zeros
    
    while True:
        delta = 0  # Initialize the maximum change in value function for this iteration
        for s in range(num_states):
            old_value = V[s]
            # Calculate the value for the current state using the Bellman equation
            V[s] = max(sum(transition_probabilities[s][a][next_s] * (rewards[s][a][next_s] + gamma * V[next_s])
                           for next_s in range(num_states))
                       for a in range(len(transition_probabilities[s])))
            delta = max(delta, abs(V[s] - old_value))  # Update the maximum change in value function
        
        # Check for convergence
        if delta < epsilon:
            break
    
    # Calculate the optimal policy based on the optimal value function
    policy = np.zeros(num_states, dtype=int)
    for s in range(num_states):
        policy[s] = np.argmax([sum(transition_probabilities[s][a][next_s] * (rewards[s][a][next_s] + gamma * V[next_s])
                                    for next_s in range(num_states))
                               for a in range(len(transition_probabilities[s]))])
    
    return policy, V

# Example usage
epsilon = 0.0001
rewards = np.array([[[-1, 0], [0, 0]], [[0, 0], [0, -1]]])  # Example rewards for each state-action pair
transition_probabilities = np.array([[[[0.9, 0.1], [0.8, 0.2]], [[0.3, 0.7], [0.6, 0.4]]],
                                     [[[0.2, 0.8], [0.5, 0.5]], [[0.7, 0.3], [0.1, 0.9]]]])
gamma = 0.9

policy, optimal_values = value_iteration(epsilon, rewards, transition_probabilities, gamma)
print("Optimal Policy:", policy)
print("Optimal Values:", optimal_values)

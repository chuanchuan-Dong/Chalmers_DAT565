import numpy as np
action_dic = {0:'N', 1:'S', 2:'W', 3:'E'}
def BellmanEquation(epsilon: float, V_init: np.array, reward: np.array, transition_prob: np.array, gamma: float):
    V_current = V_init.copy()
    while True:
        V_update = np.zeros((3,3))
        for i in range(np.shape(V_init)[0]):
            for j in range(np.shape(V_init)[1]):
                value_max = -1
                value = -1
                for next_s in [[i-1,j], [i+1,j], [i,j-1], [i,j+1]]:  #N, S, W, E
                    if 0 <= next_s[0] < np.shape(V_init)[0] and 0 <= next_s[1] < np.shape(V_init)[1]: 
                        value = ( (transition_prob[0] * (reward[next_s[0]][next_s[1]] + gamma * V_current[next_s[0]][next_s[1]])) +
                                                transition_prob[1] * (reward[i][j] + gamma * V_current[i][j]))
                    if value > value_max:
                        value_max = value
                V_update[i][j] = value_max           
        
        if np.max(abs(V_update - V_current)) < epsilon:
            policy = np.empty((3,3), dtype=str)
            for i in range(np.shape(V_init)[0]):
                for j in range(np.shape(V_init)[1]):
                    value_max = -1
                    value = -1
                    for action_index, next_s in enumerate([[i-1,j], [i+1,j], [i,j-1], [i,j+1]]):
                        if 0 <= next_s[0] < np.shape(V_init)[0] and 0 <= next_s[1] < np.shape(V_init)[1]: 
                            value = ( (transition_prob[0] * (reward[next_s[0]][next_s[1]] + gamma * V_current[next_s[0]][next_s[1]])) +
                                                    transition_prob[1] * (reward[i][j] + gamma * V_current[i][j]))
                        if value > value_max:
                            value_max = value
                            policy[i][j] = action_dic[action_index]
                        
            return V_update, policy
        else:
            V_current = V_update.copy()
            # print(V_current)

epsilon = 10
reward = np.array([[-1,1,],[0,10,0],[0,0,0]])
V_init = np.array([[0,0,0],[0,0,0],[0,0,0]])
transition_prob = np.array([0.8,0.2])
V, P = BellmanEquation(epsilon, V_init, reward, transition_prob, gamma=0.5)
print('Optimal Value:')
print(V)
print('Policy:')
print(P)
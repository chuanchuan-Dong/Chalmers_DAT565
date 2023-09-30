import numpy as np
def BellmanEquation(epsilon: float, V_init: np.array, reward: np.array, transition_prob: np.array, gamma: float):
    V_current = V_init.copy()
    while True:
        V_update = np.zeros((3,3))
        for i in range(np.shape(V_init)[0]):
            for j in range(np.shape(V_init)[1]):
                V_old = V_current[i][j]
                # print(V_old)
                # print(V_current)
                V_update[i][j] = max(
                    (transition_prob[0] * (reward[next_s[0]][next_s[1]] + gamma * V_current[next_s[0]][next_s[1]])) + 
                        transition_prob[1] * (reward[i][j] + gamma * V_old)
                                      for next_s in [[i-1,j], [i+1,j], [i,j-1], [i,j+1]]
                                      if 0 <= next_s[0] < np.shape(V_init)[0] and 0 <= next_s[1] < np.shape(V_init)[1])
                # print("vupdate:", V_update[i][j])
        if np.max(abs(V_update - V_current)) < epsilon:
            return V_update
        else:
            V_current = V_update.copy()
            print(V_current)

    





epsilon = 0.1
reward = np.array([[0,0,0],[0,10,0],[0,0,0]])
V_init = np.array([[0,0,0],[0,0,0],[0,0,0]])
transition_prob = np.array([0.8,0.2])
V = BellmanEquation(epsilon, V_init, reward, transition_prob, gamma=0.9)
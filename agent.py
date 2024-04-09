import numpy as np

class Agent():

    def __init__(self, model, **kwargs) -> None:
        self.model = model
        self.width = kwargs['WIDTH']
        self.height = kwargs['HEIGHT']
        self.epsilon = kwargs['EPSILON']

            
    def is_greedy(self, epsilon):
        return epsilon < np.random.random()
    
    def choose_action(self, state: np.ndarray):
        filtered_state = self.reshape_for_net(state)
        valid_actions = (state != -1).flatten().astype(np.int8)
        if self.is_greedy(self.epsilon):    #Exploit
            q_table = self.model.predict(filtered_state)
            valid_actions = np.ma.masked_array(q_table, valid_actions)
            return np.argmax(valid_actions), filtered_state
        else:                               #Explore
            valid_actions = np.where(valid_actions != 1)[0]
            return np.random.choice(valid_actions), filtered_state

    def reshape_for_net(self, state):
        #Channels first \_('')_/
        #couldn't print nn_input for 0s otherwise 
        filtered_state = np.zeros((9,10,20))
        for tile_num in range(0,9):
            idx1, idx2 = np.where(state == tile_num)
            filtered_state[tile_num, idx1, idx2] = 1

        return filtered_state

           

import numpy as np
from collections import deque
# from keras import backend as K
# import keras
# import tensorflow as tf
import random

# K = tf.keras.backend


class Agent():

    def __init__(self, model, **kwargs) -> None:
        self.model = model
        self.width = kwargs['WIDTH']
        self.height = kwargs['HEIGHT']
        self.discount = kwargs['DISCOUNT']
        self.epsilon = kwargs['EPSILON']
        self.epsilon_min = kwargs['EPSILON_MIN']
        self.epsilon_decay = kwargs['EPSILON_DECAY']
        self.lr = kwargs['LR']
        self.lr_min = kwargs['LR_MIN']
        self.lr_decay = kwargs['LR_DECAY']
        self.memory_limit = kwargs['AGENT_MEMORY_LIMIT']
        self.memory = deque(maxlen=self.memory_limit)
        self.batch_size = kwargs['BATCH_SIZE']
   
    def is_greedy(self):
        return self.epsilon < np.random.random()
    
    def choose_action(self, state: np.ndarray):
        filtered_state = self.reshape_for_net(state)
        valid_actions = (state != -1).flatten().astype(np.int8)
        if self.is_greedy():                #Exploit
            q_table = self.model.predict(filtered_state, verbose = 0)
            valid_actions = np.ma.masked_array(q_table, valid_actions)
            return np.argmax(valid_actions)
        else:                               #Explore
            valid_actions = np.where(valid_actions != 1)[0]
            return np.random.choice(valid_actions)

    def reshape_for_net(self, state):
        filtered_state = np.zeros((1,self.height,self.width, 9))
        for tile_num in range(0,9):
            idx1, idx2 = np.where(state == tile_num)
            filtered_state[0, idx1, idx2, tile_num] = 1
            #https://keras.io/api/layers/convolution_layers/convolution2d/#:~:text=as%20the%20input.-,data_format,-%3A%20string%2C%20either%20%22channels_last
        return filtered_state

    def add_to_memory(self, game_state):
        self.memory.append(game_state)

    def train(self):
        # batch = np.random.choice(self.memory, self.batch_size)  #Train from memory
        batch = random.sample(self.memory, self.batch_size)

        q_table_updates = []
        batch_states = []
        for _,experience in enumerate(batch):
            state, action, reward, next_state, done = experience
            filtered_state = self.reshape_for_net(state)
            current_q_table_estimate = self.model.predict(filtered_state, verbose = 0)[0]
            batch_states.append(filtered_state)

            if done: new_q = reward
            else:
                valid_actions = (next_state != -1).flatten().astype(np.int8)
                filtered_next_state = self.reshape_for_net(next_state)
                next_state_q_table_estimate = self.model.predict(filtered_next_state, verbose = 0)[0]
                valid_q_table_entries = np.ma.masked_array(next_state_q_table_estimate, valid_actions)
                new_q = reward + self.discount * np.max(valid_q_table_entries)

            # td_error = current_q_table_estimate[action] - new_q
            # td_error = np.clip(td_error, -1, 1) # Clip for stability

            current_q_table_estimate[action] = new_q
            q_table_updates.append(current_q_table_estimate)


        q_table_updates = np.array(q_table_updates)
        batch_states = np.squeeze(np.array(batch_states), axis=1) #\_('')_/

        # print(q_table_updates.shape)
        # print(batch_states.shape)
        # input('asdasd')
        print("Training batch")
        self.model.train_on_batch(batch_states, q_table_updates)

        #decay learn_rate
        # self.lr = max(self.lr_min, self.lr * self.lr_decay)
        # K.set_value(self.model.optimizer.learning_rate, self.lr) #For api 2.0

        #decay epsilon
        self.epsilon = max(self.epsilon, self.epsilon * self.epsilon_decay)



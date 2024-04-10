import gymnasium as gym
import pygame
from minesweeper_env import Minesweeper
import numpy as np
from agent import Agent
from DQN import create_dqn
from datetime import datetime

gym.register(
    id = 'Minesweeper-v0',
    entry_point = 'minesweeper_env:Minesweeper',
    kwargs = {'board_size':(10,20), 'num_mines': 15}
)

WIDTH, HEIGHT = 20, 10
NUM_MINES = 15
env = gym.make('Minesweeper-v0', board_size = (HEIGHT,WIDTH), num_mines = NUM_MINES)
env: Minesweeper

#Hyper Paramaters
DISCOUNT = 0.09 #GAMMA - Number from assignment 3
EPSILON = 0.99 # Exploration rate
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
LR = 0.0075
LR_MIN = 0.0001
LR_DECAY = 0.995

MAX_TRAINING_EPISODES = 1_000_000
AGENT_MEMORY_LIMIT = 50_000
MIN_MEMORY_LIMIT = 1_000
BATCH_SIZE = 128

CONV_FILTERS = 64

agent_kwargs = {
    'WIDTH': WIDTH,
    'HEIGHT': HEIGHT,
    'DISCOUNT': DISCOUNT,
    'EPSILON': EPSILON,
    'EPSILON_MIN': EPSILON_MIN,
    'EPSILON_DECAY': EPSILON_DECAY,
    'LR': LR,
    'LR_MIN': LR_MIN,
    'LR_DECAY': LR_DECAY,
    'MAX_TRAINING_EPISODES': MAX_TRAINING_EPISODES,
    'AGENT_MEMORY_LIMIT': AGENT_MEMORY_LIMIT,
    'BATCH_SIZE': BATCH_SIZE,
}


# Training Loop
#Create agent and shi
model = create_dqn(LR, LR_DECAY, LR_MIN, env.board_size, CONV_FILTERS)
agent = Agent(model, **agent_kwargs)
actions_taken = 0
win_history, reward_history, score_history = [], [], []

# env.reset()
# env.render()
# print(env.board)

for episode in range(MAX_TRAINING_EPISODES):
    state, _ = env.reset()
    done = False
    episode_reward = 0
    while not done:
        # print(state)
        action = agent.choose_action(state)
        game_action = action//WIDTH, action%WIDTH
        # print(game_action)
        next_state, reward, done, _ = env.step(game_action)
        agent.add_to_memory((state, action, reward, next_state, done))
        if len(agent.memory) >= MIN_MEMORY_LIMIT and actions_taken %100 == 0:
            agent.train()
        #Mby train every n steps?

        episode_reward += reward    
        state = next_state
        actions_taken += 1

    score_history.append(env.board.size - env.spaces_left())
    reward_history.append(episode_reward)
    win_history.append(1 if env.is_win() else 0)
    print(episode, reward_history[-1])

#save model
timestamp = datetime.now()
timestamp = timestamp.strftime("%d-%b-%Y-%H:%M:%S")
model.save(f'Minesweeper_{episode}-episodes_{timestamp}') #.h5


# for _ in range(50):
        
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             break
#         elif event.type == pygame.KEYDOWN:
#             if event.key == pygame.K_ESCAPE:
#                 break
#     else:
#         action = env.get_random_action()  # Random action selection
#         env.render(action)
#         obs, reward, done, _ = env.step(action)
#         print(f'Action: {action},\t Reward: {reward},\t Done: {done}')
#         # print('Done:', done)
#         if done:
#             env.reset()
#         pygame.time.wait(250)
#         continue

#     break
    


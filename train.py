import gymnasium as gym
import pygame
import numpy as np
from agent import Agent
from DQN import create_dqn

gym.register(
    id = 'Minesweeper-v0',
    entry_point = 'minesweeper_env:Minesweeper',
    kwargs = {'board_size':(10,20), 'num_mines': 15}
)

WIDTH, HEIGHT = 10, 20
NUM_MINES = 15
env = gym.make('Minesweeper-v0', board_size = (WIDTH,HEIGHT), num_mines = NUM_MINES)

EPSILON = 0.9 # Exploration rate
MAX_TRAINING_EPISODES = 1000000

agent_kwargs = {
    'WIDTH': WIDTH,
    'HEIGHT': HEIGHT,
    "EPSILON": EPSILON
}


env.reset()
env.render()

# # # Training Loop

#Create agent and shit
# model = create_dqn()
# agent = Agent(model, **agent_kwargs)

# for episode in range(MAX_TRAINING_EPISODES):
#     state = env.reset()
#     done = False
#     while not done:
#         action, filtered_state = agent.choose_action(state)
#         # action = env.get_random_action()
#         next_state, reward, done, _ = env.step(action)
#         print(f'Action: {action},\t Reward: {reward},\t Done: {done}')


for _ in range(50):
        
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            break
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                break
    else:
        action = env.get_random_action()  # Random action selection
        env.render(action)
        obs, reward, done, _ = env.step(action)
        print(f'Action: {action},\t Reward: {reward},\t Done: {done}')
        # print('Done:', done)
        if done:
            env.reset()
        pygame.time.wait(250)
        continue

    break
    


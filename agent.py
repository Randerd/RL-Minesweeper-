from minesweeper_env import Minesweeper
import gymnasium as gym
import pygame
import numpy as np

gym.register(
    id = 'Minesweeper-v0',
    entry_point = 'minesweeper_env:Minesweeper',
    kwargs = {'board_size':(10,20), 'num_mines': 15}
)

env = gym.make('Minesweeper-v0', board_size = (10,20), num_mines = 15)
env.reset()
env.render()

done = False
for i in range(50):
        
    pygame.event.get()
    action = env.get_action()  # Random action selection
    obs, reward, done, _ = env.step(action)
    env.render()
    print(f'Action: {action},\t Reward: {reward},\t Done: {done}')
    # print('Done:', done)
    if done:
        env.reset()
    pygame.time.wait(100)

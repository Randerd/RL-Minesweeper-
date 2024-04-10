import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame

BLACK = (15, 15, 15)
WHITE = (200, 200, 200)

class Minesweeper(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, board_size, num_mines, win_reward = 1, action_reward = 0.3, loss_reward = -1, guess_reward = -0.3) -> None:
        self.board_size = board_size
        self.height = board_size[0]
        self.width = board_size[1]

        self.loss_reward = loss_reward
        self.win_reward = win_reward
        self.guess_reward = guess_reward
        self.action_reward = action_reward
        self.discount = 0.9

        self.MINE = -2
        self.HIDDEN = -1
        self.num_mines = num_mines

        self.board = np.full(board_size, self.HIDDEN, dtype=int)
        self.mines = self.generate_mines()

        self.start()
        self.observation_space = spaces.Box(low=-1, high=9,
                                            shape=self.board_size, dtype=np.int32)
        self.action_space = spaces.Discrete(self.height * self.width)
        # self.action_space = gym.spaces.Dict({
        #     "x": gym.spaces.Discrete(self.width),
        #     "y": gym.spaces.Discrete(self.height),
        # })
        # self.action_space = spaces.Box(shape=self.board_size, dtype=np.int32)
        self.render_mode = "human"

        #Pygame paramaters
        self.colors  = [
            (2, 0, 253),
            (6, 124, 0),
            (251, 1, 0),
            (2, 0, 126),
            (124, 3, 5),
            (0, 129, 124),
            (0, 0, 0),
            (128, 128, 128),
        ]

        self.block_size = 30
        self.top_border = 100
        self.side_border = 16

        self.window_width = self.width * self.block_size + self.side_border * 2
        self.window_height = self.height * self.block_size + self.side_border + self.top_border

        pygame.init()
        self.window = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("Minesweeper") 
        self.clock = pygame.time.Clock()
 
    def inbounds(self, action):
        y,x = action
        return (0 <= x < self.width) & (0 <= y < self.height)

    def spaces_left(self):
        return np.count_nonzero(self.board == self.HIDDEN)

    def is_mine(self, action):
        return self.mines[action] == self.MINE
    
    def is_blank(self, action):
        return self.board[action] == 0
    
    def is_hidden(self, action):
        return self.board[action] == self.HIDDEN
    
    def is_win(self):
        return self.spaces_left() == self.num_mines

    def is_valid_move(self,action):
        return self.inbounds(action) & self.is_hidden(action)

    def reset(self, seed=None, options=None):
        self.board = np.full(self.board_size, self.HIDDEN, dtype=int)
        self.mines = self.generate_mines()
        self.start()
        # if self.render_mode == "human":
        #     self.render()
        return self.board, {}

    def generate_mines(self):
        self.mines = np.zeros(self.board_size, dtype=int)
        mines_placed = 0

        while mines_placed < self.num_mines:
            x = np.random.randint(0, self.width)
            y = np.random.randint(0, self.height)
            if self.inbounds((y,x)):
                if not self.is_mine((y,x)):
                    self.mines[y,x] = self.MINE
                    mines_placed +=1
        return self.mines

    def count_neighbour_mines(self, action):
        col,row = action
        neighbour_mines = 0
        for x in range(row-1,row+2):
            for y in range(col-1, col+2):
                if self.inbounds((y,x)):
                    if self.is_mine((y,x)):
                        neighbour_mines += 1
        return neighbour_mines

    def start(self):
        for x in range(self.width):
            for y in range(self.height):
                if self.count_neighbour_mines((y,x)) == 0:
                    self.board[y,x] = 0
                    self.show_neighbours((y,x))
                    return

    def show_neighbours(self,action):
        col,row = action
        for x in range(row-1,row+2):
            for y in range(col-1, col+2):
                if self.inbounds((y,x)):
                    if not self.is_mine((y,x)) and self.is_hidden((y,x)):
                        self.board[y,x] = self.count_neighbour_mines((y,x))
                        if self.is_blank((y,x)):
                            self.show_neighbours((y,x))

    def visible_neighbouring_cells(self, action):
        col, row = action
        for x in range(row-1,row+2):
            for y in range(col-1, col+2):
                if self.inbounds((y,x)):
                    if not self.is_hidden((y,x)):
                        return True
        return False
       
    def make_move(self, action):
        self.board[action] = self.count_neighbour_mines(action)
        if self.is_blank(action):
            self.show_neighbours(action)

    def step(self, action):
        """
        See gym.Env.step()

        Return:
        next state, reward, game over flag, info
        """

        # x,y = action
        # action = tuple(action)
        if not self.is_valid_move(action):
            # raise Exception("Invalid Action: ", action)
            return self.board, self.loss_reward, False, {}
        
        if self.is_mine(action):            #Hit mine
            return self.board, self.loss_reward, True, {}
        
        if self.visible_neighbouring_cells(action):    #Not "guess"
            reward = self.action_reward
            self.make_move(action)
        else:
            reward = self.guess_reward              #"Guess"
            self.make_move(action)
        
        if self.is_win():                   #Is win (after move)
            return self.board, self.win_reward, True, {}

        return self.board, reward, False, {}

    def get_random_action(self):
        valid_actions = (self.board == -1).flatten().astype(np.int8)
        action = self.action_space.sample(valid_actions)
        return action//self.width, action%self.width

    def drawNumber(self,num, x,y, c):
        num = str(num)
        screen_text = pygame.font.SysFont("Calibri", 16, True).render(num, True, c)
        self.window.blit(screen_text, (x,y))

    def render(self, next_action = None):
        self.window.fill(WHITE)
        for x in range(0, self.width):
            for y in range(0, self.height):
                x_pos = x*self.block_size + self.side_border
                y_pos = y*self.block_size + self.top_border
                if not self.is_hidden((y,x)) and not self.is_blank((y,x)):
                    self.drawNumber(self.board[y,x], x_pos+12, y_pos+8, self.colors[self.board[y,x]-1])

                rect = pygame.Rect(x_pos, y_pos, self.block_size, self.block_size)
                if next_action == (y,x):
                    pygame.draw.rect(self.window, (230,0,0), rect, 2)
                else:
                    pygame.draw.rect(self.window, BLACK, rect, self.board[y,x]!=0)
        
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        pygame.display.quit()
        pygame.quit()

if __name__ == "__main__":
    m = Minesweeper((10,20), 15)
    print(m.board)
    m.reset()
    print(m.board)
#     while not m.is_win():
#         x,y = input("Enter next move (y,x): ").split(",")
#         action = (int(x), int(y))
#         try:
#             state, reward, done, info = m.step((action))
#             # print(f'Reward: {reward}, Discount: {discount}')
#             m.render()
#             if done:
#                 print("GAME OVER")
#                 break
#         except Exception as e:
#             print(e)
#             m.render()
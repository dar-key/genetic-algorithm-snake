import matplotlib.pyplot as plt
import numpy as np
from helper import *
is_data_loaded, loaded_data = load_data_file(f'models/test16/data.json')

generations = []
best_scores = []
mean_scores = []

for [gen, best_sc, mean_sc] in loaded_data["History"]:
    generations.append(gen)
    best_scores.append(best_sc+2)
    mean_scores.append(mean_sc)

xpoints = np.array(generations)
ypoints = np.array(best_scores)
plt.xlabel("№ поколения")
plt.ylabel("Лучший счёт за поколение")
plt.plot(xpoints, ypoints)
plt.show()

# import random, enum, pygame as pgs
# from collections import namedtuple

# theme_colors = {
#         "background": (45, 45, 45),
#         "snake": (75, 235, 95),
#         "food": (255, 65, 65),
#         "ray_wall": (55, 55, 55),
#         "ray_food": (0, 55, 255),
#         "ray_self": (255, 55, 0),
#     }


# class Direction(enum.Enum):
#     RIGHT = 1
#     LEFT = 2
#     UP = 3
#     DOWN = 4
# TILE_SIZE = 100  # In pixels

# pg.font.init()
# font = pg.font.Font("JetBrainsMono-Regular.ttf", 14)


# vision_directions = ((1,0), (1,1), (0,1), (-1,1), (-1,0), (-1,-1), (0,-1), (1,-1))  # 8 directions
# clock_wise = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]  #  4 directions clock-wise
# Block = namedtuple('Block', ['x', 'y'])  # named tuple object for each "Block"
# class Snake:
#     def __init__(self, game):
#         self.game = game
#         self.body = []
#         self.reset()


#     def is_the_game_won(self):
#         return self.body != [] and len(self.body) == self.game.width * self.game.height // (TILE_SIZE * TILE_SIZE)


#     def reset(self):
#         self.repetitive = False
#         self.dead = False
#         self.score = 0
#         self.move_count = 0
#         start_x = random.randint(2, self.game.width//TILE_SIZE-1) * TILE_SIZE
#         start_y = random.randint(0, self.game.height//TILE_SIZE-1) * TILE_SIZE
#         self.body = [Block(start_x - TILE_SIZE*space, start_y) for space in range(3)]
#         self.head_direction = Direction.RIGHT
#         self.spawn_food()

    
#     def spawn_food(self):
#         self.food = Block(random.randrange(0, ((self.game.width) // TILE_SIZE)) * TILE_SIZE,
#                         random.randrange(0, ((self.game.height) // TILE_SIZE)) * TILE_SIZE)
#         if self.food in self.body:
#             self.spawn_food()


#     def check_collision(self, block=None, body_collision=True, wall_collision=True):
#         if block == None:
#             block = self.body[0]
#         return wall_collision and (block.x > self.game.width - TILE_SIZE or block.x < 0 or block.y > self.game.height - TILE_SIZE or block.y < 0) or body_collision and block in self.body[1:]


#     def move(self, action):
#         if self.dead:
#             return
        
#         self.move_count += 1
#         if len(action) == 3:
#             current_direction_index = clock_wise.index(self.head_direction)
#             action_case = action.index(1)

#             match action_case:
#                 case 0: # no turn, just straight
#                     new_dir = clock_wise[current_direction_index]
#                 case 1: # right
#                     next = (current_direction_index + 1) % 4
#                     new_dir = clock_wise[next]
#                 case 2: # left
#                     next = (current_direction_index - 1) % 4
#                     new_dir = clock_wise[next]
#                 case _:
#                     pass

#             self.head_direction = new_dir

#         elif len(action) == 4:
#             match action.index(1):
#                 case 0:
#                     new_dir = Direction.RIGHT
#                 case 1:
#                     new_dir = Direction.DOWN
#                 case 2:
#                     new_dir = Direction.LEFT
#                 case 3:
#                     new_dir = Direction.UP
#                 case _:
#                     pass
        
#             if not (self.head_direction == Direction.UP and new_dir == Direction.DOWN or
#                     self.head_direction == Direction.DOWN and new_dir == Direction.UP or
#                     self.head_direction == Direction.RIGHT and new_dir == Direction.LEFT or
#                     self.head_direction == Direction.LEFT and new_dir == Direction.RIGHT):
                
#                 self.head_direction = new_dir

#         x = self.body[0].x
#         y = self.body[0].y

#         if self.head_direction == Direction.RIGHT:
#             x += TILE_SIZE
#         elif self.head_direction == Direction.LEFT:
#             x -= TILE_SIZE
#         elif self.head_direction == Direction.DOWN:
#             y += TILE_SIZE
#         elif self.head_direction == Direction.UP:
#             y -= TILE_SIZE

#         self.body.insert(0, Block(x, y)) # new head
#         no_moves_left = self.move_count > len(self.body) * (self.game.width // TILE_SIZE) ** 2 * 1.2
#         if self.check_collision() or no_moves_left:
#             self.dead = True
#             if no_moves_left:
#                 self.repetitive = True
        
#         if self.body[0] == self.food:
#             self.score += 1
#             if len(self.body) >= self.game.width * self.game.height // (TILE_SIZE ** 2):
#                 print('====== The game has been completed ======')
#                 return True
#             self.spawn_food()
#         else:
#             self.body.pop()

#         return False


# class Game():
#     def __init__(self, window_height=1000, window_width=1000):
#         self.height = window_height
#         self.width = window_width
#         self.screen = pg.display.set_mode((self.width, self.height))
#         self.clock = pg.time.Clock()
    

#     def step(self):        
#         for event in pg.event.get():
#             if event.type == pg.QUIT:
#                 pg.quit()
#                 quit()
        
#         self.clock.tick(20)


#     def render(self, snake):
#         self.screen.fill(theme_colors["background"])

#         for index, block in enumerate(snake.body):
#             pg.draw.rect(self.screen, theme_colors["snake"], pg.Rect(block.x+0.1*TILE_SIZE, block.y+0.1*TILE_SIZE, TILE_SIZE*0.8, TILE_SIZE*0.8))
            
#             if index + 1 < len(snake.body):
#                 pg.draw.rect(self.screen, theme_colors["snake"], pg.Rect((block.x+snake.body[index+1].x)/2+0.1*TILE_SIZE, (block.y+snake.body[index+1].y)/2+0.1*TILE_SIZE, TILE_SIZE*0.8, TILE_SIZE*0.8))

#         pg.draw.rect(self.screen, theme_colors["food"], pg.Rect(snake.food.x+0.1*TILE_SIZE, snake.food.y+0.1*TILE_SIZE, TILE_SIZE*0.8, TILE_SIZE*0.8))
        
#         if True:
#             for each_direction in vision_directions:
#                 if each_direction[0] == 0 or each_direction[1] == 0:
#                     tile_next = 1

#                     while True:
#                         block = Block(snake.body[0].x + each_direction[0] * TILE_SIZE * tile_next, snake.body[0].y + each_direction[1] * TILE_SIZE * tile_next)
#                         if snake.check_collision(block, body_collision=False):
#                             pg.draw.line(self.screen, theme_colors["ray_wall"], (snake.body[0].x+TILE_SIZE/2, snake.body[0].y+TILE_SIZE/2), (block.x+TILE_SIZE/2, block.y+TILE_SIZE/2))
#                             text = font.render(f'{tile_next}.0', True, theme_colors.get("font", "white"))
#                             self.screen.blit(text, [(snake.body[0].x+block.x)/2, (snake.body[0].y+block.y)/2])
#                             break
#                         elif snake.check_collision(block, wall_collision=False):
#                             pg.draw.line(self.screen, theme_colors["ray_self"], (snake.body[0].x+TILE_SIZE/2, snake.body[0].y+TILE_SIZE/2), (block.x+TILE_SIZE/2, block.y+TILE_SIZE/2))
#                             text = font.render(f'{tile_next}.0', True, theme_colors.get("font", "white"))
#                             self.screen.blit(text, [(snake.body[0].x+block.x)/2, (snake.body[0].y+block.y)/2])
#                             break
#                         elif block == snake.food:
#                             pg.draw.line(self.screen, theme_colors["ray_food"], (snake.body[0].x+TILE_SIZE/2, snake.body[0].y+TILE_SIZE/2), (block.x+TILE_SIZE/2, block.y+TILE_SIZE/2))
#                             text = font.render(f'{tile_next}.0', True, theme_colors.get("font", "white"))
#                             self.screen.blit(text, [(snake.body[0].x+block.x)/2, (snake.body[0].y+block.y)/2])
#                             break
#                         tile_next += 1
        
#         pg.display.flip()



# game= Game()
# snake = Snake(game)
# snake.reset()
# snake.food = Block(6*TILE_SIZE, 6*TILE_SIZE)
# snake.body = []
# snake.body.extend([Block(4*TILE_SIZE, 6*TILE_SIZE), Block(4*TILE_SIZE, 7*TILE_SIZE), Block(3*TILE_SIZE, 7*TILE_SIZE)])
# while True:
#     game.render(snake)
#     game.step()













# # def asdf(*args):
# #     print(type(args))

# # print(asdf(1, "idk"))

# # import numpy as np  
# # from model import LinearModel
# # from helper import *
# # from game import Game, Block, Direction, Snake

# # game = Game(fps=15)
# # from agent import Agent

# # a =Agent(game)
# # a.model.load_state_dict(load_model("models/test16/best_model.pth"))

# # while True:
# #     a.decide_move(a.get_state())
# #     game.step()
# #     game.render(a.snake, {})

# #     if a.snake.dead:
# #         a.snake.reset()


# # class Agent:
# #     def __init__(self, init_model=True):
# #         self.model = LinearModel() if init_model else None 
# #         self.snake = Snake(game)

    
# #     def evaluate_fitness(self):
# #         self.fitness = 0.0005 * self.snake.move_count + 2**self.snake.score + self.snake.score ** 2 * 550 - (self.snake.score**1.2 * (0.25*self.snake.move_count)**1.3)
# #         return self.fitness


# #     def get_state(self):
# #         head = self.snake.body[0]
# #         point_l = Block(head.x - 20, head.y)
# #         point_r = Block(head.x + 20, head.y)
# #         point_u = Block(head.x, head.y - 20)
# #         point_d = Block(head.x, head.y + 20)
        
# #         dir_l = self.snake.head_direction == Direction.LEFT
# #         dir_r = self.snake.head_direction == Direction.RIGHT
# #         dir_u = self.snake.head_direction == Direction.UP
# #         dir_d = self.snake.head_direction == Direction.DOWN

# #         state = [
# #             # Danger straight
# #             (dir_r and self.snake.check_collision(point_r)) or 
# #             (dir_l and self.snake.check_collision(point_l)) or 
# #             (dir_u and self.snake.check_collision(point_u)) or 
# #             (dir_d and self.snake.check_collision(point_d)),

# #             # Danger right
# #             (dir_u and self.snake.check_collision(point_r)) or 
# #             (dir_d and self.snake.check_collision(point_l)) or 
# #             (dir_l and self.snake.check_collision(point_u)) or 
# #             (dir_r and self.snake.check_collision(point_d)),

# #             # Danger left
# #             (dir_d and self.snake.check_collision(point_r)) or 
# #             (dir_u and self.snake.check_collision(point_l)) or 
# #             (dir_r and self.snake.check_collision(point_u)) or 
# #             (dir_l and self.snake.check_collision(point_d)),
            
# #             # Move direction
# #             dir_l,
# #             dir_r,
# #             dir_u,
# #             dir_d,
            
# #             # Food location 
# #             self.snake.food.x < head.x,  # food left
# #             self.snake.food.x > head.x,  # food right
# #             self.snake.food.y < head.y,  # food up
# #             self.snake.food.y > head.y  # food down
# #             ]

# #         return np.array(state, dtype=int)
    

# #     def decide_move(self, state):
# #         action = [0, 0, 0]
# #         state = torch.tensor(state,     dtype=torch.float)
# #         self.model.eval()
# #         with torch.inference_mode():
# #             prediction = self.model(state)
# #         action_index = torch.argmax(prediction).item()
# #         action[action_index] = 1
# #         if self.snake.move(action):
# #             print("OMG")


# # agent = Agent()
# # agent.model.load_state_dict(load_model(f"models/model_completed_5x5.pth"), strict=True, assign=True)

# # while True:
# #     agent.decide_move(agent.get_state())
# #     game.step()
# #     game.render(agent.snake, None)

# #     if agent.snake.dead:
# #         agent.snake.reset()

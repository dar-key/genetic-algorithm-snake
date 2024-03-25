import torch, random, numpy as np, math
from game import Block, Direction, TILE_SIZE, Snake, vision_directions
from model import LinearModel, DEVICE, OUTPUT_SIZE
from helper import *


torch.set_default_device(DEVICE)


class Agent:
    def __init__(self, game, init_model=True):
        self.snake = Snake(game)
        self.model = LinearModel() if init_model else None
        self.fitness = 0

    
    def evaluate_fitness(self):
        x = self.snake.move_count
        y = self.snake.score
        self.fitness = (
            # y ** 1.5 * math.log10(x + 1) * 10
            # x + y ** 3 + y * 50 - (0.2 * y * x) ** 1.2
            x + 500 * y ** 1.5 / (x + 1)
        )


    def get_state(self):
        # head = self.snake.body[0]
        # point_l = Block(head.x - 20, head.y)
        # point_r = Block(head.x + 20, head.y)
        # point_u = Block(head.x, head.y - 20)
        # point_d = Block(head.x, head.y + 20)
        
        # dir_l = self.snake.head_direction == Direction.LEFT
        # dir_r = self.snake.head_direction == Direction.RIGHT
        # dir_u = self.snake.head_direction == Direction.UP
        # dir_d = self.snake.head_direction == Direction.DOWN

        # state = [
        #     # Danger straight
        #     (dir_r and self.snake.check_collision(point_r)) or 
        #     (dir_l and self.snake.check_collision(point_l)) or 
        #     (dir_u and self.snake.check_collision(point_u)) or 
        #     (dir_d and self.snake.check_collision(point_d)),

        #     # Danger right
        #     (dir_u and self.snake.check_collision(point_r)) or 
        #     (dir_d and self.snake.check_collision(point_l)) or 
        #     (dir_l and self.snake.check_collision(point_u)) or 
        #     (dir_r and self.snake.check_collision(point_d)),

        #     # Danger left
        #     (dir_d and self.snake.check_collision(point_r)) or 
        #     (dir_u and self.snake.check_collision(point_l)) or 
        #     (dir_r and self.snake.check_collision(point_u)) or 
        #     (dir_l and self.snake.check_collision(point_d)),
            
        #     # Move direction
        #     dir_l,
        #     dir_r,
        #     dir_u,
        #     dir_d,
            
        #     # Food location 
        #     self.snake.food.x < head.x,  # food left
        #     self.snake.food.x > head.x,  # food right
        #     self.snake.food.y < head.y,  # food up
        #     self.snake.food.y > head.y  # food down
        #     ]

        # return np.array(state, dtype=int)

        head = self.snake.body[0]
        tail = self.snake.body[-1]

        if tail.x < self.snake.body[-2].x:
            tail_direction = Direction.RIGHT
        elif tail.x > self.snake.body[-2].x:
            tail_direction = Direction.LEFT
        elif tail.y < self.snake.body[-2].y:
            tail_direction = Direction.DOWN
        elif tail.y > self.snake.body[-2].y:
            tail_direction = Direction.UP
        
        vision_data = []

        for each_direction in vision_directions:
            tile_next = 1
            distance_to_wall = 0
            distance_to_body = 0
            distance_to_food = 0

            while True:
                block = Block(head.x + each_direction[0] * TILE_SIZE * tile_next, head.y + each_direction[1] * TILE_SIZE * tile_next)
                if self.snake.check_collision(block, body_collision=False):
                    distance_to_wall = tile_next
                    break
                elif self.snake.check_collision(block, wall_collision=False):
                    distance_to_body = tile_next
                elif block == self.snake.food:
                    distance_to_food = tile_next
                tile_next += 1
            
            vision_data.extend([distance_to_wall, distance_to_body>0, distance_to_food>0])

        directions = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]

        return np.array([
            *vision_data, # 24 or 4*3=12
            *[self.snake.head_direction == d for d in directions], # 4
            *[tail_direction == d for d in directions], # 4
        ], dtype=int)
    

    def decide_move(self, state):
        action = np.zeros(OUTPUT_SIZE).tolist()

        state = torch.tensor(state, dtype=torch.float)
        self.model.eval()

        with torch.inference_mode():
            prediction = self.model(state)

        action_index = torch.argmax(prediction).item()
        action[action_index] = 1

        if self.snake.move(action): # Game completed by the model
            model_name = f"model_completed_{self.snake.game.width//TILE_SIZE}x{self.snake.game.height//TILE_SIZE}_{random.randint(1000,9999)}.pth"
            save_model(self.model, "models/", model_name)
            print(f"Model name: {model_name}")
            self.snake.food = Block(-TILE_SIZE, -TILE_SIZE)

            while True:
                self.snake.game.render(snake=self.snake, session_data={})
                self.snake.game.step()
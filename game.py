import random, enum, pygame as pg
from collections import namedtuple

pg.font.init()
font = pg.font.Font("JetBrainsMono-Regular.ttf", 10)

TILE_SIZE = 90  # in pixels
FRAME_RATE = 1000000000  # default frame rate (per second). It can be significantly slower because of the limitations of the device
SHOW_SNAKE_VISION = True  # Show the vision direction lines
THEMES = {  # RGB
    "Original": {
        "background": (45, 45, 45),
        "snake": (75, 235, 95),
        "food": (255, 65, 65),
        "ray_wall": (55, 55, 55),
        "ray_food": (0, 55, 255),
        "ray_self": (255, 55, 0),
    },

    "White": {
        "background": (255, 255, 255),
        "snake": (45, 216, 129),
        "food": (239, 35, 60),
        "ray_wall": (54, 53, 55),
        "ray_food": (114, 90, 193),
        "ray_self": (226, 177, 177),
        "font": (52, 62, 61),
    },
    
    "Tropical Indigo": {
        "background": (58, 64, 90),
        "snake": (153, 178, 221),
        "food": (233, 175, 163),
        "ray_wall": (155, 155, 155),
        "ray_food": (255, 255, 255),
        "ray_self": (233, 100, 90),
    },

    "Emerald": {
        "background": (21, 49, 49),
        "snake": (69, 203, 133),
        "food": (106, 127, 219),
        "ray_wall": (34, 65, 65),
        "ray_food": (87, 226, 229),
        "ray_self": (224, 141, 172),
    },
}


theme_colors = THEMES["Original"]  # chosen theme


class Direction(enum.Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


vision_directions = ((1,0),(0,1),(-1,0),(0,-1))#((1,0), (1,1), (0,1), (-1,1), (-1,0), (-1,-1), (0,-1), (1,-1))  # 8 directions
clock_wise = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]  #  4 directions clock-wise
Block = namedtuple('Block', ['x', 'y'])  # named tuple object for each "Block"


class Snake:
    def __init__(self, game):
        self.game = game
        self.body = []
        self.reset()


    def is_the_game_won(self):
        return self.body != [] and len(self.body) == self.game.width * self.game.height // (TILE_SIZE * TILE_SIZE)


    def reset(self):
        self.repetitive = False
        self.dead = False
        self.score = 0
        self.move_count = 0
        start_x = random.randint(2, self.game.width//TILE_SIZE-1) * TILE_SIZE
        start_y = random.randint(0, self.game.height//TILE_SIZE-1) * TILE_SIZE
        self.body = [Block(start_x - TILE_SIZE*space, start_y) for space in range(3)]
        self.head_direction = Direction.RIGHT
        self.spawn_food()

    
    def spawn_food(self):
        self.food = Block(random.randrange(0, ((self.game.width) // TILE_SIZE)) * TILE_SIZE,
                        random.randrange(0, ((self.game.height) // TILE_SIZE)) * TILE_SIZE)
        if self.food in self.body:
            self.spawn_food()


    def check_collision(self, block=None, body_collision=True, wall_collision=True):
        if block == None:
            block = self.body[0]
        return wall_collision and (block.x > self.game.width - TILE_SIZE or block.x < 0 or block.y > self.game.height - TILE_SIZE or block.y < 0) or body_collision and block in self.body[1:]


    def move(self, action):
        if self.dead:
            return
        
        self.move_count += 1
        if len(action) == 3:
            current_direction_index = clock_wise.index(self.head_direction)
            action_case = action.index(1)

            match action_case:
                case 0: # no turn, just straight
                    new_dir = clock_wise[current_direction_index]
                case 1: # right
                    next = (current_direction_index + 1) % 4
                    new_dir = clock_wise[next]
                case 2: # left
                    next = (current_direction_index - 1) % 4
                    new_dir = clock_wise[next]
                case _:
                    pass

            self.head_direction = new_dir

        elif len(action) == 4:
            match action.index(1):
                case 0:
                    new_dir = Direction.RIGHT
                case 1:
                    new_dir = Direction.DOWN
                case 2:
                    new_dir = Direction.LEFT
                case 3:
                    new_dir = Direction.UP
                case _:
                    pass
        
            if not (self.head_direction == Direction.UP and new_dir == Direction.DOWN or
                    self.head_direction == Direction.DOWN and new_dir == Direction.UP or
                    self.head_direction == Direction.RIGHT and new_dir == Direction.LEFT or
                    self.head_direction == Direction.LEFT and new_dir == Direction.RIGHT):
                
                self.head_direction = new_dir

        x = self.body[0].x
        y = self.body[0].y

        if self.head_direction == Direction.RIGHT:
            x += TILE_SIZE
        elif self.head_direction == Direction.LEFT:
            x -= TILE_SIZE
        elif self.head_direction == Direction.DOWN:
            y += TILE_SIZE
        elif self.head_direction == Direction.UP:
            y -= TILE_SIZE

        self.body.insert(0, Block(x, y)) # new head
        no_moves_left = self.move_count > len(self.body) * (self.game.width // TILE_SIZE) ** 2 * 1.2
        if self.check_collision() or no_moves_left:
            self.dead = True
            if no_moves_left:
                self.repetitive = True
        
        if self.body[0] == self.food:
            self.score += 1
            if len(self.body) >= self.game.width * self.game.height // (TILE_SIZE ** 2):
                print('====== The game has been completed ======')
                return True
            self.spawn_food()
        else:
            self.body.pop()

        return False



class Game():
    def __init__(self, window_height=900, window_width=900, fps=0):
        self.fps = fps
        if fps <= 0:
            self.fps = FRAME_RATE
        self.height = window_height
        self.width = window_width
        self.screen = pg.display.set_mode((self.width, self.height))
        self.clock = pg.time.Clock()
    

    def step(self):        
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                quit()
        
        self.clock.tick(self.fps)


    def render(self, snake: Snake, session_data: dict):
        self.screen.fill(theme_colors["background"])

        for index, block in enumerate(snake.body):
            pg.draw.rect(self.screen, theme_colors["snake"], pg.Rect(block.x+0.1*TILE_SIZE, block.y+0.1*TILE_SIZE, TILE_SIZE*0.8, TILE_SIZE*0.8))
            
            if index + 1 < len(snake.body):
                pg.draw.rect(self.screen, theme_colors["snake"], pg.Rect((block.x+snake.body[index+1].x)/2+0.1*TILE_SIZE, (block.y+snake.body[index+1].y)/2+0.1*TILE_SIZE, TILE_SIZE*0.8, TILE_SIZE*0.8))

        pg.draw.rect(self.screen, theme_colors["food"], pg.Rect(snake.food.x+0.1*TILE_SIZE, snake.food.y+0.1*TILE_SIZE, TILE_SIZE*0.8, TILE_SIZE*0.8))
        
        if SHOW_SNAKE_VISION:
            for each_direction in vision_directions:
                tile_next = 1

                while True:
                    block = Block(snake.body[0].x + each_direction[0] * TILE_SIZE * tile_next, snake.body[0].y + each_direction[1] * TILE_SIZE * tile_next)
                    if snake.check_collision(block, body_collision=False):
                        pg.draw.line(self.screen, theme_colors["ray_wall"], (snake.body[0].x+TILE_SIZE/2, snake.body[0].y+TILE_SIZE/2), (block.x+TILE_SIZE/2, block.y+TILE_SIZE/2))
                        break
                    elif snake.check_collision(block, wall_collision=False):
                        pg.draw.line(self.screen, theme_colors["ray_self"], (snake.body[0].x+TILE_SIZE/2, snake.body[0].y+TILE_SIZE/2), (block.x+TILE_SIZE/2, block.y+TILE_SIZE/2))
                        break
                    elif block == snake.food:
                        pg.draw.line(self.screen, theme_colors["ray_food"], (snake.body[0].x+TILE_SIZE/2, snake.body[0].y+TILE_SIZE/2), (block.x+TILE_SIZE/2, block.y+TILE_SIZE/2))
                        break
                    tile_next += 1

        text_to_render = ""

        for name, value in session_data.items():
            if type(value) != dict and type(value) != list and type(value) != tuple:
                text_to_render += f'{name}: {value:,.1f} | '

        text = font.render(text_to_render, True, theme_colors.get("font", "white"))
        self.screen.blit(text, [0, 0])
        pg.display.flip()


import torch, random, os, time
from agent import Agent
from model import LinearModel, DEVICE
from helper import *
from game import Game
from algorithm.crossover import SBX, SPBX
from algorithm.mutation  import gaussian_mutation
from algorithm.selection import roulette_wheel_selection, elitism_selection
from typing import Tuple


torch.set_default_device(DEVICE)


POPULATION_SIZE = 1000      # POPULATION_SIZE is even number
MUTATION_RATE = 0.05       # gaussian mutation probability (dynamic or static)
MUTATION_SCALE = 0.2       # gaussian mutation scale
MUTATION_TYPE = "static"  # options: ["static", "dynamic"]
NUM_PARENTS = 500          # POPULATION_SIZE >= NUM_PARENTS


active_mutation_rate = MUTATION_RATE
active_mutation_scale = MUTATION_SCALE


def custom_crossover(parent1: LinearModel, parent2: LinearModel) -> Tuple[LinearModel, LinearModel]:
    return SBX(parent1, parent2) if random.random() > 0.5 else SPBX(parent1, parent2)


def custom_mutation(*models: Tuple[LinearModel]) -> None:
    for model in models:
        gaussian_mutation(model, active_mutation_rate, active_mutation_scale)


if __name__ == '__main__':
    NAME_OF_THE_SESSION = input('Save folder: models/')
    
    session_path = f'models/{NAME_OF_THE_SESSION}'
    population = []
    game = Game(fps=0)


    # Population initialization
    session_data = {
        "Generation": 1,
        "AT Best Score": 0,
        "AT Best Fitness": -torch.inf,
        "Best Score": 0,
        "Mean Score": 0,
        "Time": 0,
        "Alive": 0,
        "Mean Fitness": -torch.inf,
        "Start Time": time.time(),
        "History": [],
    }

    is_data_loaded = False
    

    if os.path.exists(f"models/{NAME_OF_THE_SESSION}"):
        is_data_loaded, loaded_data = load_data_file(f'{session_path}/data.json')
    

    if is_data_loaded:
        for key, value in loaded_data.items():
            session_data[key] = value
            
        session_data["Start Time"] = time.time() - session_data["Time"]

        best_model_state_dict = load_model(f"{session_path}/best_model.pth")
        
        for i in range(POPULATION_SIZE):
            new_agent = Agent(game)
            loaded_model = load_model(f"{session_path}/model_{i}.pth")

            if loaded_model is not None:
                new_agent.model.load_state_dict(loaded_model)
                new_agent.model.eval()
                population.append(new_agent)
    else:
        create_dir(f"models/{NAME_OF_THE_SESSION}")

        for i in range(POPULATION_SIZE):
            new_agent = Agent(game)
            population.append(new_agent)


    # Genetic algorithm loop
    while True:
        session_data["Time"] = time.time() - session_data["Start Time"]
        session_data["Alive"] = 0

        for each_agent in population:
            if not each_agent.snake.dead:
                session_data["Alive"] += 1
                agent_state = each_agent.get_state()
                each_agent.decide_move(agent_state)
        

        if session_data["Alive"] == 0:
            session_data["Generation"] += 1

            # calculation
            fitness_sum = 0
            score_sum = 0
            agent_best_score = population[0] # max(population, key=lambda agent: agent.snake.score)
            agent_best_fitness = population[0] # max(population, key=lambda agent: agent.fitness)

            for agent in population:
                agent.evaluate_fitness()

                if agent.snake.score > agent_best_score.snake.score:
                    agent_best_score = agent

                if agent.fitness > agent_best_fitness.fitness:
                    agent_best_fitness = agent

                fitness_sum += agent.fitness
                score_sum += agent.snake.score
            
            session_data["Mean Fitness"] = fitness_sum / POPULATION_SIZE
            session_data["Mean Score"] = score_sum / POPULATION_SIZE

            # selection
            potential_parents = elitism_selection(list(filter(lambda agent: not agent.snake.repetitive, population)), NUM_PARENTS)

            # set new best
            for i, agent in enumerate(population):
                save_model(agent.model, session_path, f'model_{i}.pth')

            if MUTATION_TYPE == "dynamic":
                if (session_data["Best Score"] - agent_best_score.snake.score) > 3: # old > new  |  very bad
                    active_mutation_rate = MUTATION_RATE * 1.5
                    active_mutation_scale = MUTATION_SCALE * 1.2
                elif (session_data["Best Score"] - agent_best_score.snake.score) > 0: # old > new  |  bad
                    active_mutation_rate = MUTATION_RATE * 1.2
                    active_mutation_scale = MUTATION_SCALE * 1.1
                elif (session_data["Best Score"] - agent_best_score.snake.score) < 0: # old < new  |  good
                    active_mutation_rate = MUTATION_RATE * 0.8
                    active_mutation_scale = MUTATION_SCALE * 0.9
                elif (session_data["Best Score"] - agent_best_score.snake.score) < -3: # old < new  | very good
                    active_mutation_rate = MUTATION_RATE * 0.4
                    active_mutation_scale = MUTATION_SCALE * 0.6

            session_data["Best Score"] = agent_best_score.snake.score
            session_data["History"].append((session_data["Generation"], session_data["Best Score"], session_data["Mean Score"], session_data["Mean Fitness"]))
            if session_data["Best Score"] > session_data["AT Best Score"]:
                session_data["AT Best Score"] = agent_best_score.snake.score
            
            if agent_best_fitness.fitness > session_data["AT Best Fitness"]:
                save_model(agent_best_fitness.model, session_path, f'best_model.pth')
                
                best_model_state_dict = agent_best_fitness.model.state_dict()
                session_data["AT Best Fitness"] = agent_best_fitness.fitness

            save_data_file(f'{session_path}/data.json', session_data)

            if random.random() < 0.08:
                best_agent = Agent(game)
                best_agent.model.load_state_dict(best_model_state_dict)
                best_agent.fitness = session_data["AT Best Fitness"]
                potential_parents.append(best_agent)

            random.shuffle(potential_parents)

            # NEW GENERATION
            for i in range(0, POPULATION_SIZE, 2):
                population[i].snake.reset()
                population[i+1].snake.reset()
                parent_1, parent_2 = roulette_wheel_selection(potential_parents, 2)
                child_1, child_2 = custom_crossover(parent_1, parent_2)
                custom_mutation(child_1, child_2)
                population[i].model = child_1
                population[i+1].model = child_2
        else:
            population.sort(key=lambda agent: agent.snake.score, reverse=True)

            for agent in population:
                if not agent.snake.dead:
                    chosen_snake_to_render = agent.snake
                    break

            game.render(snake=chosen_snake_to_render, session_data=session_data)

        game.step()

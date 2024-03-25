import random


def elitism_selection(population: list, num_selection: int) -> list:
    sorted_by_fit = sorted(population, key = lambda agent: agent.fitness, reverse=True)
    if num_selection < len(population):
        return sorted_by_fit[:num_selection]
    else:
        return sorted_by_fit


def roulette_wheel_selection(population: list, num_selection: int) -> list:
    selection = []
    wheel = sum(agent.fitness for agent in population)
    
    for _ in range(num_selection):
        pick = random.uniform(0, wheel)
        current = 0
        for agent in population:
            current += agent.fitness
            if current > pick:
                selection.append(agent.model)
                break

    return selection
import random, torch
from model import LinearModel, DEVICE


torch.set_default_device(DEVICE)


def SBX(parent1: LinearModel, parent2: LinearModel, eta: float = 100):
    child1 = LinearModel()
    child2 = LinearModel()

    with torch.inference_mode():
        for (p1_param, p2_param, c1_param, c2_param) in zip(parent1.parameters(), parent2.parameters(), child1.parameters(), child2.parameters()):
            rand = torch.rand(p1_param.data.shape)
            gamma = torch.empty(p1_param.data.shape)
            first_half = rand <= 0.5
            second_half = rand > 0.5
            gamma[first_half] = (2 * rand[first_half]) ** (1.0 / (eta + 1))
            gamma[second_half] = (1 / (2 * (1 - rand[second_half]))) ** (1 / (eta + 1))
            c1_param.data = 0.5 * ((1 + gamma) * p1_param.data + (1 - gamma) * p2_param.data)
            c2_param.data = 0.5 * ((1 - gamma) * p1_param.data + (1 + gamma) * p2_param.data)

    return child1, child2


def Some_Func(parent1: LinearModel, parent2: LinearModel, alpha: float = 0.5):
    child_model = LinearModel()
    with torch.no_grad():
        for child_param, parent1_param, parent2_param in zip(child_model.parameters(), parent1.parameters(), parent2.parameters()):
            child_param.data = (1 - alpha) * parent1_param.data + alpha * parent2_param.data
    return child_model


def SPBX(parent1: LinearModel, parent2: LinearModel, major='r'):
    child1 = LinearModel()
    child2 = LinearModel()
    
    with torch.inference_mode():
        for (p1_param, p2_param, c1_param, c2_param) in zip(parent1.parameters(), parent2.parameters(), child1.parameters(), child2.parameters()):
            shape1 = p1_param.shape
            shape2 = p2_param.shape
            crossover_point = random.randint(0, min(shape1[0], shape2[0]))

            if major.lower() == 'r':
                c1_param.data[:crossover_point] = p2_param.data[:crossover_point]
                c1_param.data[crossover_point:] = p1_param.data[crossover_point:]

                c2_param.data[:crossover_point] = p1_param.data[:crossover_point]
                c2_param.data[crossover_point:] = p2_param.data[crossover_point:]

            elif major.lower() == 'c':
                c1_param.data = torch.cat((p2_param.data[:crossover_point], p1_param.data[crossover_point:]))
                c2_param.data = torch.cat((p1_param.data[:crossover_point], p2_param.data[crossover_point:]))

    return child1, child1



# def SBX_old(parent1: LinearModel, parent2: LinearModel, eta=12, pc=0.9):
#     child1 = LinearModel()
#     child2 = LinearModel()
    
#     with torch.inference_mode():
#         for param1, param2, child1_param, child2_param in zip(parent1.parameters(), parent2.parameters(), child1.parameters(), child2.parameters()):
#             if random.random() < pc:
#                 u = random.random()

#                 if u <= 0.5:
#                     beta = (2 * u)**(1.0 / (eta + 1))
#                 else:
#                     beta = (1 / (2 * (1 - u)))**(1.0 / (eta + 1))
                
#                 child1_param.data = 0.5 * (((1 + beta) * param1.data) + ((1 - beta) * param2.data))
#                 child2_param.data = 0.5 * (((1 - beta) * param1.data) + ((1 + beta) * param2.data))
#             else:
#                 child1_param.data = param1.data.clone()
#                 child2_param.data = param2.data.clone()

#     return child1, child2





# def SBX(parent1, parent2, eta: float):
#     rand = np.random.random(parent1.shape)
#     gamma = np.empty(parent1.shape)
#     gamma[rand <= 0.5] = (2 * rand[rand <= 0.5]) ** (1.0 / (eta + 1))  # First case of equation 9.11
#     gamma[rand > 0.5] = (1.0 / (2.0 * (1.0 - rand[rand > 0.5]))) ** (1.0 / (eta + 1))  # Second case

#     # Calculate Child 1 chromosome (Eq. 9.9)
#     chromosome1 = 0.5 * ((1 + gamma)*parent1 + (1 - gamma)*parent2)
#     # Calculate Child 2 chromosome (Eq. 9.10)
#     chromosome2 = 0.5 * ((1 - gamma)*parent1 + (1 + gamma)*parent2)

#     return chromosome1, chromosome2


# def SPBX(parent1, parent2):
#     offspring1 = parent1.copy()
#     offspring2 = parent2.copy()

#     rows, cols = parent2.shape
#     row = np.random.randint(0, rows)
#     col = np.random.randint(0, cols)

#     if major.lower() == 'r':
#         offspring1[:row, :] = parent2[:row, :]
#         offspring2[:row, :] = parent1[:row, :]

#         offspring1[row, :col+1] = parent2[row, :col+1]
#         offspring2[row, :col+1] = parent1[row, :col+1]
#     elif major.lower() == 'c':
#         offspring1[:, :col] = parent2[:, :col]
#         offspring2[:, :col] = parent1[:, :col]

#         offspring1[:row+1, col] = parent2[:row+1, col]
#         offspring2[:row+1, col] = parent1[:row+1, col]

#     return offspring1, offspring2


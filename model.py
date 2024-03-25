import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


INPUT_LAYER = 20
HIDDEN_LAYERS = [20, 12]
OUTPUT_SIZE = 4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_device(DEVICE)


class LinearModel(nn.Module):
    def __init__(self, input_layer: int = INPUT_LAYER, hidden_layers: Tuple[int] | List[int] = HIDDEN_LAYERS, output_layer: int = OUTPUT_SIZE):
        super().__init__()
        self.hidden_layers_number = len(hidden_layers)
        self.input_layer = nn.Linear(in_features=input_layer, out_features=hidden_layers[0])
        if self.hidden_layers_number == 2: # Neural network: [input, hidden_1, hidden_2, output];  else: [input, hidden, output]
            self.hidden_layer = nn.Linear(in_features=hidden_layers[0], out_features=hidden_layers[-1]) 
        self.output_layer = nn.Linear(in_features=hidden_layers[-1], out_features=output_layer)


    def forward(self, x: torch.Tensor):
        if self.hidden_layers_number == 2:
            x = F.relu(self.input_layer(x))
            x = F.relu(self.hidden_layer(x))
            x = F.sigmoid(self.output_layer(x))
        else:
            x = F.relu(self.input_layer(x))
            x = self.output_layer(x)
        return x


    # def _save(self, data: list=None, data_file_index='data', model_name='model.pth'):
    #     folder_path = Path('models')
    #     folder_path.mkdir(parents=True, exist_ok=True)
    #     model_path = folder_path / model_name
    #     torch.save(obj=self.state_dict(), f=model_path)

    #     data_file = f"{model_name[:model_name.rfind('.')]}-{data_file_index}.json"
    #     with open(folder_path/data_file, 'w') as data_file:
    #         json.dump(data, data_file)


    # def _load(self, default=None, data_file_index='data', model_name='model.pth'):
    #     if os.path.exists(f'models/{model_name}'):
    #         data_file = f"{model_name[:model_name.rfind('.')]}-{data_file_index}.json"
    #         folder_path = Path('models')
    #         folder_path.mkdir(parents=True, exist_ok=True)

    #         with open(folder_path/data_file, 'r') as data_file:
    #             data = json.load(data_file)
    #         # print("DATA = ", data)
            
    #         model_path = folder_path / model_name
    #         self.load_state_dict(torch.load(f=model_path))
    #         return data, True
        
    #     return default, False


# class ModelTrainer:
#     def __init__(self, model, lr, dr):
#         self.dr = dr
#         self.model = model
#         self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#         self.criterion = nn.MSELoss()

#     def train(self, state, action, reward, next_state, done):
#         state = torch.tensor(np.array(state), dtype=torch.float)
#         next_state = torch.tensor(np.array(next_state), dtype=torch.float)
#         action = torch.tensor(action, dtype=torch.long)
#         reward = torch.tensor(reward, dtype=torch.float)

#         if len(state.shape) == 1:
#             state = torch.unsqueeze(state, 0)
#             next_state = torch.unsqueeze(next_state, 0)
#             action = torch.unsqueeze(action, 0)
#             reward = torch.unsqueeze(reward, 0)
#             done = (done, )

#         # 1: predicted Q values with current state
#         self.model.train()
#         pred = self.model(state)

#         target = pred.clone()
#         for idx in range(len(done)):
#             Q_new = reward[idx]
#             if not done[idx]:
#                 Q_new = reward[idx] + self.dr * torch.max(self.model(next_state[idx]))

#             target[idx][torch.argmax(action[idx]).item()] = Q_new
    
#         # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
#         # pred.clone()
#         # preds[argmax(action)] = Q_new
#         self.optimizer.zero_grad()
#         loss = self.criterion(target, pred)
#         loss.backward()

#         self.optimizer.step()

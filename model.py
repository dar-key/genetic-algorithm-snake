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

import torch, os, json
from pathlib import Path
from model import LinearModel, DEVICE


torch.set_default_device(DEVICE)


def create_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def save_data_file(path: str, data: dict | list) -> None:
    file_path = Path(path)
    with open(file_path, 'w') as file:
        json.dump(data, file)


def load_data_file(path: str) -> any:
    if os.path.exists(path):
        file_path = Path(path)

        with open(file_path, 'r') as file:
            data = json.load(file)
        
        return True, data
    
    return False, None


def save_model(model: LinearModel, path: str, file_name: str) -> None:
    folder_path = Path(path)
    folder_path.mkdir(parents=True, exist_ok=True)
    file = folder_path / file_name
    torch.save(obj=model.state_dict(), f=file)
    

def load_model(path: str) -> any:
    if os.path.exists(path):
        file = Path(path)
        return torch.load(f=file)
    else:
        print("No such path: ", path)
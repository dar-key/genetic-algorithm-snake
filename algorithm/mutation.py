import torch
from model import LinearModel, DEVICE


torch.set_default_device(DEVICE)


def gaussian_mutation(model: LinearModel, rate: float, scale: float = None) -> None:

    with torch.inference_mode():
        for model_param in model.parameters():
            genes_to_mutate = torch.rand(model_param.data.shape) < rate

            gaussian_distribution = torch.normal(0, 1, device=DEVICE, size=model_param.data.shape)

            if scale:
                gaussian_distribution[genes_to_mutate] *= scale

            model_param.data[genes_to_mutate] += gaussian_distribution[genes_to_mutate]
    

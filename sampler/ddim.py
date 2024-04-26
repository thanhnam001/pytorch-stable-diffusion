import torch
import numpy as np

class DDIMSampler:
    def __init__(self,
                 generator: torch.Generator,
                 num_training_step: int=1000,
                 beta_start: float=0.00085,
                 beta_end: float = 0.0120,
                 device: str = 'cpu') -> None:
        self.betas = (torch.linspace(beta_start ** ))
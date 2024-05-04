import torch
import numpy as np
from sampler.base_sampler import Sampler

class DDIMSampler(Sampler):
    def __init__(self,
                 generator: torch.Generator,
                 num_training_steps: int = 1000,
                 beta_start: float = 0.00085,
                 beta_end: float = 0.012,
                 ddim_discretize: str = 'uniform',
                 ddim_eta: float = 0.,
                 device: str = 'cpu'):
        super().__init__(generator, num_training_steps, beta_start, beta_end, device)
        self.ddim_eta = ddim_eta
        self.set_inference_timestep(num_training_steps, ddim_discretize=ddim_discretize)
    
    def set_inference_timestep(self, num_inference_steps=50, ddim_discretize: str = 'uniform'):
        self.num_inference_steps = num_inference_steps
        if ddim_discretize=='uniform':
            step_ratio = self.num_train_timesteps // num_inference_steps
            self.timesteps = torch.from_numpy((np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64))
        elif ddim_discretize=='quadratic':
            print(((np.linspace(0, np.sqrt(self.num_train_timesteps * .8), num_inference_steps)) ** 2).astype(int)[::-1])
            self.timesteps = torch.from_numpy(((np.linspace(0, np.sqrt(self.num_train_timesteps * .8), num_inference_steps)) ** 2).astype(int)[::-1].copy())
        else:
            raise NotImplementedError(ddim_discretize)
        
    def step(self, timestep: int, latents: torch.Tensor, model_output: torch.Tensor):
        # timestep 
        # latents
        # model_output (eps_t)
        t = timestep
        prev_t = self._get_previous_timestep(t)
        print(prev_t)
        bs = model_output.shape[0]
        
        # Compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[t] # \bar{alpha}_t
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one # \bar{alpha}_{t-1}
        beta_prod_t = 1 - alpha_prod_t # \bar{beta}_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev # \bar{beta}_{t-1}
        alpha_t = alpha_prod_t / alpha_prod_t_prev # alpha_t
        alpha_t_prev = self.alphas[prev_t] # alpha_{t-1}
        beta_t = 1 - alpha_t # beta_t
        sqrt_alpha_t = torch.sqrt(alpha_t)
        
        ddim_sigma_t = self.ddim_eta * torch.sqrt((1 - alpha_t)/(1 - alpha_t_prev)) * torch.sqrt(1 - alpha_t/alpha_t_prev) # sigma_{t}
        sqrt_1m_alpha_t = torch.sqrt(1 - alpha_t)
        
        x_0 = (latents - sqrt_1m_alpha_t * model_output)/ sqrt_alpha_t
        dir_xt = torch.sqrt(1. - alpha_t_prev - ddim_sigma_t**2) * model_output
        noise = torch.randn(model_output.shape, generator=self.generator, device=model_output.device, dtype=model_output.dtype)
        
        x_prev = torch.sqrt(alpha_t_prev) * x_0 + dir_xt + ddim_sigma_t * noise
        
        return x_prev
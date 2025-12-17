import torch

from models.diffusion.scheduler_registry import get_noise_schedule
from models.diffusion.timestep_sampler import get_timestep_sampler


def test_noise_schedules_monotone_betas():
    for name in ["linear", "cosine", "logsnr_laplace"]:
        schedule = get_noise_schedule(name, timesteps=100)
        betas = schedule.betas
        assert betas.dim() == 1 and betas.numel() == 100
        assert torch.all(betas > 0)
        assert torch.all(betas < 1)
        assert torch.all(schedule.alpha_bar <= 1)
        assert torch.all(schedule.alpha_bar >= 0)


def test_timestep_samplers_basic_range():
    schedule = get_noise_schedule("linear", timesteps=50)
    for name in ["uniform", "logsnr_importance", "adaptive_region"]:
        sampler = get_timestep_sampler(name, schedule=schedule, device=torch.device("cpu"))
        idx = sampler.sample(batch_size=16)
        assert idx.shape == (16,)
        assert idx.dtype == torch.long
        assert torch.all(idx >= 0) and torch.all(idx < schedule.timesteps)

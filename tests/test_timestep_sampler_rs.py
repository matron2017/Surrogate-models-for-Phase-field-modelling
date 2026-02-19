import pytest
import torch

from models.diffusion.scheduler_registry import BrownianBridgeSchedule, UniDBCosineSchedule
from models.diffusion.timestep_sampler import UniformTimestepSampler


def test_uniform_timestep_sampler_default_range():
    schedule = BrownianBridgeSchedule(timesteps=16, sigma=1.0)
    sampler = UniformTimestepSampler(schedule=schedule)
    t = sampler.sample(batch_size=1024)
    assert int(t.min().item()) >= 0
    assert int(t.max().item()) <= 15


def test_uniform_timestep_sampler_open_interval_for_bridge():
    schedule = BrownianBridgeSchedule(timesteps=16, sigma=1.0)
    sampler = UniformTimestepSampler(schedule=schedule, t_min=1, t_max=15)
    t = sampler.sample(batch_size=1024)
    # t_max is exclusive, so this is [1, 14]
    assert int(t.min().item()) >= 1
    assert int(t.max().item()) <= 14


def test_uniform_timestep_sampler_includes_terminal_step_for_unidb():
    schedule = UniDBCosineSchedule(timesteps=20)
    sampler = UniformTimestepSampler(schedule=schedule, t_min=1, t_max=20)
    t = sampler.sample(batch_size=8192)
    assert int(t.min().item()) >= 1
    # UniDB uses inclusive t_max semantics, so T must be reachable.
    assert int(t.max().item()) == 20


@pytest.mark.parametrize(
    "t_min,t_max,errmsg",
    [
        (-1, None, "t_min must be >= 0"),
        (0, 17, "t_max must be <="),
        (5, 5, "t_max must be >"),
        (6, 5, "t_max must be >"),
    ],
)
def test_uniform_timestep_sampler_invalid_bounds(t_min, t_max, errmsg):
    schedule = BrownianBridgeSchedule(timesteps=16, sigma=1.0)
    with pytest.raises(ValueError, match=errmsg):
        UniformTimestepSampler(schedule=schedule, t_min=t_min, t_max=t_max)

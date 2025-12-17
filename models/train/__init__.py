"""Training utilities for models backbones and tasks."""

# Re-export common builders to keep import paths concise
from models.train.loss_registry import build_surrogate_loss, build_diffusion_loss  # noqa: F401

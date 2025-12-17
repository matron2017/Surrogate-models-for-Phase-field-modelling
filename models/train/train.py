"""
Lightweight wrapper to expose the training entrypoint as `models.train.train`.
Required by tests that import `from models.train import train as train_main`.
"""

from models.train.core import train as train_main

main = train_main.main


__all__ = ["main"]

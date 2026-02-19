# Tests

Automated correctness checks for active code paths.

Run all:
- `PYTHONPATH=models /scratch/project_2008261/physics_ml/bin/python3.11 -m pytest tests -q`

Fast smoke:
- `PYTHONPATH=models /scratch/project_2008261/physics_ml/bin/python3.11 -m pytest tests/test_backbones_rs.py -q`

#!/bin/bash
PYTHONPATH=. /home/arthur/research/mldnn/.venv/bin/python3 experiments/exp1_deterministic.py
PYTHONPATH=. /home/arthur/research/mldnn/.venv/bin/python3 experiments/exp2_linear_diffusion.py
PYTHONPATH=. /home/arthur/research/mldnn/.venv/bin/python3 experiments/exp3_nonlinear_drift.py
PYTHONPATH=. /home/arthur/research/mldnn/.venv/bin/python3 experiments/exp4_fgbm.py
PYTHONPATH=. /home/arthur/research/mldnn/.venv/bin/python3 experiments/exp5_sine_cosine.py

#!/bin/sh
source venv/bin/activate

# PYOPENCL_CTX=0 ./main.py
CUDA_HOME=/opt/cuda ./main.py

#!/bin/sh
cd HPC/exercise03/
make power_gpu
cd ../..
sbatch HPC/exercise03/power_gpu.sh

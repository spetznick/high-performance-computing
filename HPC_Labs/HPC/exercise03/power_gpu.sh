#!/bin/sh
#SBATCH --job-name="power_gpu"
#SBATCH --time=00:05:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --account=Education-EEMCS-Courses-IN4049TU
#SBATCH --output=HPC/out/%x_%j.out
#SBATCH --error=HPC/out/%x_%j.err

module load 2023r1 cuda/11.6
cd HPC/exercise03/
srun power_gpu.x --size 4000

#!/bin/sh
#
#SBATCH --job-name="helloworld"
#SBATCH --partition=compute
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --account=Education-EEMCS-Courses-IN4049TU
#SBATCH --output=HPC/out/%x.out
#SBATCH --error=HPC/out/%x.err

module load 2022r2
module load openmpi

cd HPC/intro/

mpicc helloworld.c -o helloworld.x
srun helloworld.x
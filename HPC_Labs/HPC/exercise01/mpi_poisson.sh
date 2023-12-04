#!/bin/sh
#
#SBATCH --job-name="mpi_poisson"
#SBATCH --partition=compute
#SBATCH --time=0:01:00
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --account=Education-EEMCS-Courses-IN4049TU
#SBATCH --output=HPC/out/%x_%j.out
#SBATCH --error=HPC/out/%x_%j.err

module load 2022r2
module load openmpi

cd HPC/exercise01/

mpicc mpi_poisson.c -o mpi_poisson.x -lm
srun mpi_poisson.x 4 1 1.95

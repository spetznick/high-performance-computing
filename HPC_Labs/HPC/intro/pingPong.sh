#!/bin/sh
#
#SBATCH --job-name="pingpong"
#SBATCH --partition=compute
#SBATCH --time=1:00:00
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --account=Education-EEMCS-Courses-IN4049TU
#SBATCH --output=HPC/out/%x_%j.out
#SBATCH --error=HPC/out/%x_%j.err
##SBATCH --nodes=1

# TODO modify the number of process (nodes) from 1 to 2

module load 2022r2
module load openmpi

cd HPC/intro/

mpicc pingPong.c -o pingPong.x
srun pingPong.x

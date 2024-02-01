#!/bin/sh
#
#SBATCH --job-name="mpi_fempois"
#SBATCH --partition=compute
#SBATCH --time=0:02:00
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --account=Education-EEMCS-Courses-IN4049TU
#SBATCH --output=HPC/out/%x_%j.out
#SBATCH --error=HPC/out/%x_%j.err

module load 2022r2
module load openmpi

cd HPC/exercise02/

gcc -o GridDist GridDist.c -lm
mpicc MPI_Fempois.c -o MPI_Fempois.x -lm
./GridDist 2 2 100 100

srun MPI_Fempois.x

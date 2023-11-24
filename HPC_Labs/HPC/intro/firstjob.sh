#!/bin/sh
#SBATCH –-job-name=job_name
#SBATCH --partition=compute
#SBATCH –-account=research-eemcs-diam
#SBATCH --time=01:00:00
#SBATCH --ntasks=24
#SBATCH --cpus-per-task=1
#SBATCH –-mem-per-cpu=1GB
module load 2022r2
module load openmpi
srun ./executable > output.log
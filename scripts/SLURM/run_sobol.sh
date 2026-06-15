#!/bin/bash
#SBATCH --job-name=sobol_sampling
#SBATCH --error=error.log
#SBATCH --nodes=1
#SBATCH --partition=sharing,short,west
#SBATCH --mem=10Gb
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=36
#SBATCH --array=0-200%24

# takes in the settings.yaml that contains the temperature
# the SLURM_ARRAY_TASK_ID becomes the chunk index for running sobol sampling
python -m ezuq.sobol $1 $SLURM_ARRAY_TASK_ID

#!/bin/bash
#SBATCH --job-name=hybrid_cv
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=6GB
#SBATCH --time=02:00:00
#SBATCH --array=0-4             # optional, if you want multiple independent experiments
#SBATCH -o /u/slurm_output/hybrid_cv-%A_%a.out

module load proxy
module load julia/1.11.4

export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK}
echo "Running Julia with $JULIA_NUM_THREADS threads on SLURM task ${SLURM_ARRAY_TASK_ID}"

# optional: each array index can pass to Julia (e.g. different random seeds)
id=$SLURM_ARRAY_TASK_ID

julia --project --heap-size-hint=16G HybridBD.jl $id



#!/bin/bash
#SBATCH --pty -p skylake
#! Number of required nodes (can be omitted in most cases)
#SBATCH -N 128
#! Number of tasks
#SBATCH --ntasks-per-node=1
#! Number of cores per task (use for multithreaded jobs, by default 1)
#SBATCH -c 1
#!How much wallclock time will be required (HH:MM:SS)
#SBATCH --time=12:00:00

num_cores=$(lscpu | awk '/^Core\(s\) per socket:/ {print $4}')
echo "Number of cores: $num_cores"

cores_to_use=2
dist=$((num_cores / cores_to_use))

seeds=($(seq 1000 $((1000 + num_cores * 30 - 1))))

for seed in "${seeds[@]}"; do
    julia -p "$dist" --project=. beta_plane_SL_restart_dist.jl "$seed" &
done

wait

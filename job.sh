#!/bin/bash
#SBATCH --partition=long                      # Ask for unkillable job
#SBATCH --cpus-per-task=6                    # Ask for 2 CPUs
#SBATCH --gres=gpu:1                         # Ask for 1 GPU
#SBATCH --mem=8G                             # Ask for 10 GB of RAM
#SBATCH --time=2:00:00                        # The job will run for 3 hours
#SBATCH -o /network/scratch/m/moksh.jain/logs/torchseqmoo-%j.out  # Write the log on tmp1

module load python/3.7 cuda/11.1
export PYTHONUNBUFFERED=1

source /home/mila/m/moksh.jain/lambo/env/bin/activate
export PATH=$PATH:$(pwd)
# pip install ml
cd ~/mo_gfn
pkill -9 wandb
python main.py "$@"
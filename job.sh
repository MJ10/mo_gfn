#!/bin/bash
#SBATCH --partition=long                      # Ask for unkillable job
#SBATCH --cpus-per-task=6                    # Ask for 2 CPUs
#SBATCH --gres=gpu:rtx8000:1                         # Ask for 1 GPU
#SBATCH --mem=8G                             # Ask for 10 GB of RAM
#SBATCH --time=6:00:00                        # The job will run for 3 hours
#SBATCH -o /network/scratch/m/moksh.jain/logs/torchseqmoo-%j.out  # Write the log on tmp1

module load python/3.9 cuda/11.7
export PYTHONUNBUFFERED=1

cd $SLURM_TMPDIR
virtualenv env
source env/bin/activate

export PATH=$PATH:$(pwd)

cd ~/mo_gfn
pip install -e .

python main.py "$@"

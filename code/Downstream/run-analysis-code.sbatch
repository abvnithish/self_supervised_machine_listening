#!/bin/bash

#SBATCH --verbose
#SBATCH --job-name=ssml_tsa_$1
#SBATCH --output=slurm_ssml_tsa_$1.out
#SBATCH --error=slurm_ssml_tsa_$1.err

#SBATCH --nodes=1
#SBATCH --mem=6GB
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00

#SBATCH --mail-type=END
#SBATCH --mail-user=jk6373@nyu.edu

module purge
source $HOME/pyenv/ssml/bin/activate
module load cuda/10.1.105
module load cudnn/10.0v7.6.2.24

cd $HOME

python -W ignore $HOME/self_supervised_machine_listening/code/downstream/trng_size_analysis.py $1
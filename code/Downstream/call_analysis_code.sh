#!/bin/bash
                              
for i in 1 2 5
do 
    sbatch -J ssml_tsa_$i -o $SCRATCH/slurm_ssml_tsa_$i.out -e $SCRATCH/slurm_ssml_tsa_$i.err $HOME/self_supervised_machine_listening/code/downstream/run-analysis-code.sbatch $i
done
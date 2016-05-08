#!/bin/bash
#
#***
#*** "#SBATCH" lines must come before any non-blank, non-comment lines ***
#***
#
# 3 nodes, 4 CPUs per node (total 12 CPUs), wall clock time of 5 hours
#
#SBATCH -N 1                  ## Node count
#SBATCH --gres=gpu:1 --ntasks-per-node=4 -N 1 ## 4 processors per node
#SBATCH --ntasks-per-node=4   ## Processors per node
#SBATCH -t 24:00:00            ## Walltime
#
# send mail if the process fails
#SBATCH --mail-type=fail
# Remember to set your email address here instead of nobody
#SBATCH --mail-user=kl9@princeton.edu
#

module load mpi

srun ./program
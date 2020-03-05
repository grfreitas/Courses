#!/bin/bash
##
## Copyright (C) 2009-2017 VersatusHPC, Inc.
##
## ntasks-per-node = quantidade de nucleos por node
#SBATCH --ntasks=1
##
## time = qtde de horas necessaria
#SBATCH --time=360:00:00
#
## Nome do job . Aparece na saida do comando 'qstat' .
## E recomendado, mas nao necesssario, que o nome do job
## seja o mesmo que o nome do arquivo de input
#SBATCH --job-name=GAN
#
#SBATCH --gres=gpu:1

## Configura o ambiente de execucao do software.
#module load libraries/mpich/3.2-gnu-5.3
module load softwares/anaconda2/5.0-intel-2018.0
module load cuda/8.0

source activate GAN-GPU

srun python train.py gpu=1
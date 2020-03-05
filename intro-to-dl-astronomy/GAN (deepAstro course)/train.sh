#!/bin/bash
##
## Copyright (C) 2009-2017 VersatusHPC, Inc.
##
## ntasks-per-node = quantidade de nucleos por node
#SBATCH --ntasks=40
#SBATCH --partition=cosmoobs

## time = qtde de horas necessaria
#SBATCH --time=360:00:00
#
## Configura o envio de e-mail quando o job for cancelado/finalizado.
## Substitua "root" por seu endereco de e-mail.
### SBATCH --mail-type=BEGIN,FAIL,END
### SBATCH --mail-user=patrickschubert32@hotmail.com
#
## Nome do job . Aparece na saida do comando 'qstat' .
## E recomendado, mas nao necesssario, que o nome do job
## seja o mesmo que o nome do arquivo de input
#SBATCH --job-name=GAN
#
#SBATCH --gres=gpu:1

echo -e "\n## Job iniciado em $(date +'%d-%m-%Y as %T') #####################\n"

# Informacoes do job impressos no arquivo de saida.
echo -e "\n## Jobs ativos de $USER: \n"
squeue -a --user=$USER
echo -e "\n## Node de execucao do job:         $(hostname -s) \n"
echo -e "\n## Numero de tarefas para este job: $SLURM_NTASKS \n"

#########################################
##-------  Inicio do trabalho     ----- #
#########################################

## Configura o ambiente de execucao do software.
#module load libraries/mpich/3.2-gnu-5.3
module load softwares/anaconda3/5.0-intel-2018.0
module load cuda/8.0

## Informacoes sobre o ambiente de execucao impressos no arquivo de saida.
echo -e "\n## Diretorio de submissao do job:   $SLURM_SUBMIT_DIR \n"

source activate GAN-GPU
cd $SLURM_SUBMIT_DIR

## Execucao do software. Renomeie os arquivos .fits conforme o necessario
python train.py gpu=1 > log-3.out

echo -e "\n## Job finalizado em $(date +'%d-%m-%Y as %T') ###################\n"
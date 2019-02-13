#!/bin/bash
#!/bin/bash
#SBATCH --job-name=GAN
#SBATCH --output=train.out
#SBATCH --ntasks=64
source activate GAN
cd /home/treinamento/aluno08/GAN/
python train.py --input fitsdata/fits_train --fwhm 1.4 --sig 1.2 --mode 0

#!/bin/bash -x
#SBATCH --output=out_%A_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=1:00:00
#SBATCH --mem=256GB
#SBATCH --gres=gpu:2
#SBATCH --constraint="a100"
#SBATCH --job-name=arborclip.sbatch
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=<email>

#move to main folder before running

module purge;

#debug flags
echo $SLURM_JOB_NAME
export NCCL_ASYNC_ERROR_HANDLING=1


#run command
srun --accel-bind=v \
    /bin/bash run-singularity.bash \
    /bin/bash -c \
    'export PYTHONPATH="$PYTHONPATH:$PWD/src"; python -u src/training/main.py --save-frequency 1 --train-data './data/group/000000.tar' --val-data './data/group/000001.tar' --dataset-type 'webdataset' --pretrained 'openai' --text_type 'random' --dataset-resampled --warmup 1000 --batch-size 400 --train-num-samples 962 --accum-freq 1 --epochs 100 --workers 8 --model ViT-B-16 --log-every-n-steps 1 --lr 1e-4 --seed 42 --local-loss --gather-with-grad --grad-checkpointing --logs '../storage/log/''

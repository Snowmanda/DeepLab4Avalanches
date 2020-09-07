#!/usr/bin/env bash


# dataset hyperparameters
train_root_dir="/cluster/work/igp_psr/bartonp/slf_avalanches/2018"

# training hyperparameters
gpus=-1 # set this under BSUB command for cluster
default_root_dir="/cluster/scratch//bartonp"
row_log_interval=5
log_save_interval=20

#Model hyperparameters
lr=1e-3


# Parameters for bsub command
#BSUB -n 4
#BSUB -W 10
#BSUB -R "rusage[ngpus_excl_p=1]"


python -m trainer.train \
--train_root_dir $train_root_dir \
--gpus $gpus \
--default_root_dir $default_root_dir \
--row_log_interval $row_log_interval \
--log_save_interval $log_save_interval \
--lr $lr \
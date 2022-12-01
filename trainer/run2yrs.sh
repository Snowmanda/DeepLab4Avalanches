#!/usr/bin/env bash

export PYTHONPATH=$PWD
	
exp_name="Training0.25_1"
# specify path if retraining is desired "/path/to/the/checkpoint/file.ckpt"
checkpoint="/home/elyas/Desktop/SpotTheAvalanche/DroneData/Checkpoints/epoch=12-step=32850.ckpt"
resume_training=True

# Dataset hyperparameters
train_root_dir="/home/elyas/Desktop/SpotTheAvalanche/Training/Arelen/Arelen_1/"
train_ava_file="Annotation_20220203_04_SalezerArelen_lv95.shp"
train_region_file="AOI_Arelen.shp"
val_root_dir="/home/elyas/Desktop/SpotTheAvalanche/Training/Huereli/Huereli_1"
val_ava_file="Annotation_20210201_Huereli.shp"
val_region_file="AOI_Huereli.shp"

train_root_dir2="/home/elyas/Desktop/SpotTheAvalanche/Training/Breaman/Breaman_1/"
train_ava_file2="Annotation_20190119_BraemaN_lv95.shp"
train_region_file2="AOI_braema.shp"
val_root_dir2="/home/elyas/Desktop/SpotTheAvalanche/Training/Huereli/Huereli_1"
val_ava_file2="Annotation_20210201_Huereli.shp"
val_region_file2="AOI_Huereli.shp"
# optional test with avalanche point validation data
#val_gt_file=''
# optional test with avalanche point validation data
#val_gt_file2=''
dem_dir='/home/elyas/Desktop/SpotTheAvalanche/DroneData/DEM/davos_dem_25cm_lv95.tif'
tile_size=512
aval_certainty=1
# bands to be used from the optical data
bands='1 2 3'
num_workers=20
# mean for the bands specified
means="180 180 180"
# std for the bands specified
stds="80 80 80"

# Data augmentation
hflip_p=0.5
rand_rotation=180

# Training hyperparameters
# bce, weighted_bce, focal, soft_dice or bce_edges
loss=weighted_bce
seed=42
deterministic=False
gpus=1
batch_size=4
batch_augm=2
accumulate_grad_batches=2
max_epochs=20
val_check_interval=1.0
log_every_n_steps=200
flush_logs_every_n_steps=200
accelerator="ddp"
sync_batchnorm=True
log_dir="/home/elyas/Desktop/SpotTheAvalanche/DroneData/Checkpoints"
benchmark=True
# Model hyperparameters

# avanet or deeplabv3+ or deeplab or sa_unet or mask_rcnn
model='avanet'
# adapted_resnetxx, resnetxx or any other model from pytorch_segmentation_models submodule
backbone='adapted_resnet34'
# avanet or deeplabv3+ or deeplab or sa_unet or mask_rcnn
decoder='avanet'
# adam or sgd
optimiser="adam"
lr=1e-4
# multistep or plateau
lr_scheduler='multistep'
scheduler_steps="10"
scheduler_gamma=0.25
momentum=0.9
weight_decay=0.0
#In_Channel=3
in_channels=4
train_viz_interval=2000
val_viz_interval=1
val_viz_idx=4

# Avanet options
avanet_rep_stride_with_dil=True
avanet_px_per_iter=4
decoder_out_ch=512
decoder_dspf_ch="64 128 256"
decoder_rates="4 8 12"

python -m trainer.train \
--exp_name $exp_name \
--date "$(date +"%d.%m.%y")" \
--time "$(date +"%T")" \
--checkpoint "$checkpoint" \
--resume_training $resume_training \
--train_root_dir $train_root_dir \
--train_ava_file $train_ava_file \
--train_region_file $train_region_file \
--val_root_dir $val_root_dir \
--val_ava_file $val_ava_file \
--val_region_file $val_region_file \
--train_root_dir2 $train_root_dir2 \
--train_ava_file2 $train_ava_file2 \
--train_region_file2 $train_region_file2 \
--val_root_dir2 $val_root_dir2 \
--val_ava_file2 $val_ava_file2 \
--val_region_file2 $val_region_file2 \
--dem_dir $dem_dir \
--tile_size $tile_size \
--aval_certainty $aval_certainty \
--bands $bands \
--num_workers $num_workers \
--means $means \
--stds $stds \
--hflip_p $hflip_p \
--rand_rotation $rand_rotation \
--loss $loss \
--seed $seed \
--deterministic $deterministic \
--gpus $gpus \
--batch_size $batch_size \
--batch_augm $batch_augm \
--accumulate_grad_batches $accumulate_grad_batches \
--max_epochs $max_epochs \
--val_check_interval $val_check_interval \
--log_every_n_steps $log_every_n_steps \
--flush_logs_every_n_steps $flush_logs_every_n_steps \
--accelerator $accelerator \
--sync_batchnorm $sync_batchnorm \
--log_dir $log_dir \
--benchmark $benchmark \
--model $model \
--backbone $backbone \
--decoder $decoder \
--optimiser $optimiser \
--lr $lr \
--momentum $momentum \
--weight_decay $weight_decay \
--in_channels $in_channels \
--train_viz_interval $train_viz_interval \
--val_viz_interval $val_viz_interval \
--val_viz_idx $val_viz_idx \
--scheduler_gamma $scheduler_gamma \
--scheduler_steps $scheduler_steps \
--lr_scheduler $lr_scheduler \
--avanet_rep_stride_with_dil $avanet_rep_stride_with_dil \
--avanet_px_per_iter $avanet_px_per_iter \
--decoder_out_ch $decoder_out_ch \
--decoder_dspf_ch $decoder_dspf_ch \
--decoder_rates $decoder_rates \

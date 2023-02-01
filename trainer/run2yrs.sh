#!/usr/bin/env bash

export PYTHONPATH=$PWD
	
exp_name="Training0.25_5_IncreasedSamplePoints"
# specify path if retraining is desired "/path/to/the/checkpoint/file.ckpt"
checkpoint="/home/elyas/Desktop/SpotTheAvalanche/DroneData/Checkpoints/epoch=12-step=32850.ckpt"
resume_training=True
# Dataset hyperparameters

#Arelen
train_root_dir="/home/elyas/Desktop/SpotTheAvalanche/Training/Arelen/Arelen_1/"
train_ava_file="Annotation_20220203_04_SalezerArelen_lv95.shp"
train_region_file="AOI_Arelen.shp"
means1="120.95 130.46 152.56"
stds1="91 84.76 71.46"

#Braeman
train_root_dir2="/home/elyas/Desktop/SpotTheAvalanche/Training/Breaman/Breaman_1/"
train_ava_file2="Annotation_20190119_BraemaN_lv95.shp"
train_region_file2="AOI_braema.shp"
means2="186.14 194.32 209.41"
stds2="74.62 66.75 55.99"

#Dorfberg
train_root_dir3="/home/elyas/Desktop/SpotTheAvalanche/Training/Dorfberg/Dorfberg_1/"
train_ava_file3="Annotation_20220319_Dorfberg_lv95.shp"
train_region_file3="AOI_Dorfberg.shp"
means3="190.6 190.49 192.18"
stds3="30.4 59.74 58.95"

train_root_dir4="/home/elyas/Desktop/SpotTheAvalanche/Training/Dorfberg/Dorfberg_2/"
train_ava_file4="Annotation_20210225_Dorfberg.shp"
train_region_file4="AOI_Dorfberg.shp"
means4="200.88 201.22 202.48"
stds4="71.13 71.2 71.09"

train_root_dir5="/home/elyas/Desktop/SpotTheAvalanche/Training/Dorfberg/Dorfberg_3/"
train_ava_file5="Annotation_20210225_Dorfberg_Flug2.shp"
train_region_file5="AOI_Dorfberg.shp"
means5="196.81 198.59 203.88"
stds5="64.8 62.73 59.12"

#Gaudergrat
train_root_dir6="/home/elyas/Desktop/SpotTheAvalanche/Training/Gaudergrat/Gaudergrat_1/"
train_ava_file6="Annotation_20211105_Gaudergrat_lv95.shp"
train_region_file6="AOI_Gaudergrat.shp"
means6="202.59 206.7 215.84"
stds6="39.82 37.05 30.43"

train_root_dir7="/home/elyas/Desktop/SpotTheAvalanche/Training/Gaudergrat/Gaudergrat_2/"
train_ava_file7="Annotation_20220205_Gaudergrat_lv95.shp"
train_region_file7="AOI_Gaudergrat.shp"
means7="164.54 170.04 183.6"
stds7="74.38 69.30 57.5"

#Gruenboedeli
train_root_dir8="/home/elyas/Desktop/SpotTheAvalanche/Training/Gruenboedeli/Gruenboedeli_1/"
train_ava_file8="Annotation_20220103_Gruenboedeli_lv95.shp"
train_region_file8="AOI_Gruenboedeli.shp"
means8="169.44 171.9 175.98"
stds8="91.65 90.44 90.79"

train_root_dir9="/home/elyas/Desktop/SpotTheAvalanche/Training/Gruenboedeli/Gruenboedeli_2/"
train_ava_file9="Annotation_20220217_Gruenboedeli_lv95.shp"
train_region_file9="AOI_Gruenboedeli.shp"
means9="178.84 181.3 185.94"
stds9="86.49 85.38 85.29"

train_root_dir10="/home/elyas/Desktop/SpotTheAvalanche/Training/Latschuelfurgga/Latschuelfurgga_Small/"
train_ava_file10="Annotation_20210204_Latschuelfurgga.shp"
train_region_file10="AOI_Latschuelfurgga_small.shp"
means10="174.78 179.4 192.06"
stds10="33.85 32.48 28.033"


val_root_dir="/home/elyas/Desktop/SpotTheAvalanche/Training/Huereli/Huereli_1"
val_ava_file="Annotation_20210201_Huereli.shp"
val_region_file="AOI_Huereli.shp"

val_root_dir2="/home/elyas/Desktop/SpotTheAvalanche/Training/Huereli/Huereli_1"
val_ava_file2="Annotation_20210201_Huereli.shp"
val_region_file2="AOI_Huereli.shp"
means0="193.84 197.05 205.91"
stds0="57.44 55.16 48.76"
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

# std for the bands specified


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
--train_region_file3 $train_region_file3 \
--train_root_dir3 $train_root_dir3 \
--train_ava_file3 $train_ava_file3 \
--train_region_file4 $train_region_file4 \
--train_root_dir4 $train_root_dir4 \
--train_ava_file4 $train_ava_file4 \
--train_region_file5 $train_region_file5 \
--train_root_dir5 $train_root_dir5 \
--train_ava_file5 $train_ava_file5 \
--train_region_file6 $train_region_file6 \
--train_root_dir6 $train_root_dir6 \
--train_ava_file6 $train_ava_file6 \
--train_region_file7 $train_region_file7 \
--train_root_dir7 $train_root_dir7 \
--train_ava_file7 $train_ava_file7 \
--train_region_file8 $train_region_file8 \
--train_root_dir8 $train_root_dir8 \
--train_ava_file8 $train_ava_file8 \
--train_region_file9 $train_region_file9 \
--train_root_dir9 $train_root_dir9 \
--train_ava_file9 $train_ava_file9 \
--train_region_file10 $train_region_file10 \
--train_root_dir10 $train_root_dir10 \
--train_ava_file10 $train_ava_file10 \
--dem_dir $dem_dir \
--tile_size $tile_size \
--aval_certainty $aval_certainty \
--bands $bands \
--num_workers $num_workers \
--means1 $means1 \
--stds1 $stds1 \
--means2 $means2 \
--stds2 $stds2 \
--means3 $means3 \
--stds3 $stds3 \
--means4 $means4 \
--stds4 $stds4 \
--means5 $means5 \
--stds5 $stds5 \
--means6 $means6 \
--stds6 $stds6 \
--means7 $means7 \
--stds7 $stds7 \
--means8 $means8 \
--stds8 $stds8 \
--means9 $means9 \
--stds9 $stds9 \
--means10 $means0 \
--stds10 $stds0 \
--means0 $means0 \
--stds0 $stds0 \
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

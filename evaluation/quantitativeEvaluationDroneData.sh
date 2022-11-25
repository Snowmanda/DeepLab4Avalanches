#!/usr/bin/env bash

python -m evaluation.quantitativeEvaluationDroneData \
--output_dir '/home/elyas/Desktop/SpotTheAvalanche/DroneData/Output_0.25' \
--img_dir 'Drone_imagery_0.25' \
--root_dir '/home/elyas/Desktop/SpotTheAvalanche/DroneData/Regions' \
--experiment_folder '/home/elyas/Desktop/SpotTheAvalanche/DeepLab4Avalanches/experiments' \
--dem_path '/home/elyas/Desktop/SpotTheAvalanche/DroneData/davos_dem_25cm_lv95.tif' \
--checkpoint '/home/elyas/Desktop/SpotTheAvalanche/DroneData/Checkpoints/epoch=12-step=32850.ckpt' \



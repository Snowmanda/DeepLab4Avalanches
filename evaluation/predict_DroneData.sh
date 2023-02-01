#!/usr/bin/env bash

python -m evaluation.predict_region_DroneData \
--img_dir 'Drone_imagery_0.25' \
--root_dir '/home/elyas/Desktop/SpotTheAvalanche/DroneData/Regions' \
--dem_path '/home/elyas/Desktop/SpotTheAvalanche/DroneData/davos_dem_25cm_lv95.tif' \
--output_path '/home/elyas/Desktop/SpotTheAvalanche/DroneData/Output_AllPatches_Holdout' \
--checkpoint '/home/elyas/Desktop/SpotTheAvalanche/DroneData/Checkpoints/epoch=17-step=33600.ckpt' \
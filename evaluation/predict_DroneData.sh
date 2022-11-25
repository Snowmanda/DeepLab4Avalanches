#!/usr/bin/env bash

python -m evaluation.predict_region_DroneData \
--img_dir 'Drone_imagery_1.5' \
--root_dir '/home/elyas/Desktop/SpotTheAvalanche/DroneData/Regions_1.5' \
--dem_path '/home/elyas/Desktop/SpotTheAvalanche/DroneData/davos_dem_25cm_lv95.tif' \
--output_path '/home/elyas/Desktop/SpotTheAvalanche/DroneData/Output_1.5' \
--checkpoint '/home/elyas/Desktop/SpotTheAvalanche/DroneData/Checkpoints/epoch=12-step=32850.ckpt' \
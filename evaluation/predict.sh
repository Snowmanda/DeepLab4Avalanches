#!/usr/bin/env bash

python -m evaluation.predict_region \
--image_dir '/home/elyas/Desktop/SpotTheAvalanche/DroneData/Regions/Gaudergrat/Gaudergrat_1/Drone_imagery' \
--dem_path '/home/elyas/Desktop/SpotTheAvalanche/DroneData/davos_dem_25cm_lv95.tif' \
--region_file '/home/elyas/Desktop/SpotTheAvalanche/DroneData/Regions/Gaudergrat/Gaudergrat_1/AOI_Gaudergrat.shp' \
--output_path '/home/elyas/Desktop/SpotTheAvalanche/DroneData/Regions/Gaudergrat/Gaudergrat_1/Output_Gaudergrat_1' \
--checkpoint '/home/elyas/Desktop/SpotTheAvalanche/DroneData/Checkpoints/epoch=12-step=32850.ckpt' \
--aval_path '' \
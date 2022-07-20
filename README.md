#DeepLab4Avalanches

This repository contains code for automatic avalanche segmentation in optical satellite imagery using deep learning. For an explanation of the concepts, ideas and evaluation, see publication below. Please use the following citation when using the code for a publication:

Hafner, E. D., Barton, P., Daudt, R. C., Wegner, J. D., Schindler, K., and Bühler, Y.: Automated avalanche mapping from SPOT 6/7 satellite imagery: results, evaluation, potential and limitations, The Cryosphere Discuss. [preprint], https://doi.org/10.5194/tc-2022-80, in review, 2022.

More documentation is provided in the form of docstrings throughout the code.
Structure

    datasets: code for reading satellite images and ground truth labels in the form of pytorch datasets
    evaluation: various scripts for running quantitative and qualitative evaluation
    experiments: pytorch lightning module for running experiments in an easily configurable way
    modeling: model architecture and parts
    trainer: contains training scripts
    utils: various utilities for data manipulation, visualization, etc.

Installation

    Create virtual environment
    Install gdal
    Change to root directory
    pip install -r requirements.txt
    pip install -e .

Training

Training can be run with bash run1yr.sh or bash run2yrs.sh script. Training can be customised and hyperparameters set by passing the corresponding flags. All available options can be shown with:

python trainer/train.py --help
Prediction

To automatically predict avalanches on new satellite images run bash predict_region.sh script. This creates a 1 band GeoTiff with probability values ranging from 0-1, where is certainly no avalanche, and 1 is certainly an avalanche. The predictions can be thresholded to get a binary map of predictions, with their certainty determined by the threshold.

The following flags need to be set:

    --image_dir: directory containing all satellite images
    --dem_path: path to the DEM if needed
    --region_file: a shapefile specifiying which region to predict over
    --output_path: the complete path and file name in which to store predictions. This will be a GeoTiff

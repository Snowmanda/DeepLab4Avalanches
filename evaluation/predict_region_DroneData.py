""" This script can be used to run a model across an entire region and save predictions to a GeoTiff.

To see all configuration options, run the script with the --help flag.
The script can be run having set all relevant flags of paths to the input images, output directory and region file. The
region file is needed to define which area needs to be predicted and is expected to be a shapefile.

Predictions will be written to a GeoTiff stored under the output directory specified with the flags. Predictions are
not thresholded but raw floats between 0 and 1 representing the probability of an avalanche.

A ground truth avalanche file may also be included with the --aval_path flag if metrics are to be computed.
"""

import os
import numpy as np
from osgeo import gdal, osr
import geopandas as gpd
from tqdm import tqdm
from argparse import ArgumentParser
from experiments.easy_experiment import EasyExperiment
from datasets.avalanche_dataset_grid import AvalancheDatasetGrid
from torch.utils.data import DataLoader
from utils.losses import crop_to_center, get_precision_recall_f1, soft_dice
from utils import data_utils
import errno


def create_raster(region_file, output_path, tile_size, pixel_w):
    """ Create the output raster such that it covers the entire region to be predicted and uses EPSG 2056 coordinate
    system.
    """
    region = gpd.read_file(region_file)
    minx, miny, maxx, maxy = region.buffer(tile_size, join_style=2).total_bounds
    x_size = int((maxx - minx) // pixel_w)
    y_size = int((maxy - miny) // pixel_w)

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(2056)

    driver = gdal.GetDriverByName('GTiff')
    out_raster = driver.Create(os.path.join(output_path), x_size, y_size, 1, gdal.GDT_Float32)
    driver = None
    out_raster.SetGeoTransform((minx, pixel_w, 0, maxy, 0, -pixel_w))
    out_raster.SetProjection(srs.ExportToWkt())
    band_out = out_raster.GetRasterBand(1)
    band_out.SetNoDataValue(0)
    return out_raster


def write_prediction(band, array, coords, border, ulx, uly, pixel_w):
    """ Write the predictions of a patch to the raster.
    :param band: band of the output raster to be written to
    :param array: numpy array of predictions to be written to raster
    :param coords: tuple of (x,y) coordinates of the top left corner of the patch
    :param ulx: coordinate of the most left pixel in the raster
    :param uly: coordinate of the top most pixel in the raster
    :param pixel_w: Pixel width or spatial resolution
    """
    xoff = int((coords[0] - ulx) / pixel_w + border)
    yoff = int((uly - coords[1]) / pixel_w + border)

    band.WriteArray(array.squeeze().cpu().numpy(), xoff=xoff, yoff=yoff)
    band.FlushCache()  # Write cached raster to file


def compute_metrics(metrics, y, y_hat):
    """ Compute metrics for a patch """
    y_mask = data_utils.labels_to_mask(y)
    pred = y_hat.round()

    precision, recall, f1 = get_precision_recall_f1(y, pred)
    precision_back, recall_back, f1_back = get_precision_recall_f1(y_mask == 0, pred == 0)
    metrics['dice_score'].append(soft_dice(y_mask, y_hat).item())
    metrics['precision'].append(precision.item())
    metrics['recall'].append(recall.item())
    metrics['f1'].append(f1.item())
    metrics['precision_back'].append(precision_back.item())
    metrics['recall_back'].append(recall_back.item())
    metrics['f1_back'].append(f1_back.item())


def print_metrics(metrics):
    for key, val in metrics.items():
        metrics[key] = np.nanmean(np.array(val)).item()
    print(metrics)


def init_metrics():
    metrics = {'dice_score': [],
               'precision': [],
               'recall': [],
               'f1': [],
               'precision_back': [],
               'recall_back': [],
               'f1_back': []}
    return metrics

def get_all_means_stds(InputRegion):
    """Returns for a region mean 1, 2 ,3 and std 1, 2 ,3 in a list"""
    RegionImageData = {}
    RegionImageData["Arelen_1"] = [120.95, 130.46, 152.56, 91, 84.76, 71.46]
    RegionImageData["Salezer_1"] = [206,209.88, 215.82, 67.63, 54.8, 124.3]
    RegionImageData["Breaman_1"] = [186.14, 194.32, 209.41, 74.62, 66.75, 55.99]
    RegionImageData["Dorfberg_1"] = [190.6, 190.49, 192.18, 71.13, 71.2, 71.09]
    RegionImageData["Dorfberg_2"] = [200.88, 201.22, 202.48, 30.4, 59.74, 58.95]
    RegionImageData["Dorfberg_3"] = [196.81, 198.59, 203.88, 64.8, 62.73, 59.12]
    RegionImageData["Gaudergrat_1"] = [202.59, 206.7, 215.84, 39.82, 37.05, 30.43]
    RegionImageData["Gaudergrat_2"] = [164.54, 170.04, 183.6, 74.38, 69.30, 57.5]
    RegionImageData["Gruenboedeli_1"] = [169.44, 171.9, 175.98, 91.65, 90.44, 90.79]
    RegionImageData["Gruenboedeli_2"] = [178.84, 181.3, 185.94, 86.49, 85.38, 85.29]
    RegionImageData["Huereli_1"] = [193.84, 197.05, 205.91, 57.44, 55.16, 48.76]
    RegionImageData["Latschuelfurgga_large"] = [165.27, 168.22, 176.14, 71.25, 70.28, 69.36]
    RegionImageData["Latschuelfurgga_Small"] = [174.78, 179.4, 192.06, 33.85, 32.48, 28.033]
    
    for region in RegionImageData:
        if InputRegion == region:
            return RegionImageData[region]
    return print("Error: Missing Region Mean/Std Data for ", InputRegion)
            

def get_all_region_paths(root_dir,ImgFileName):
    """returns path to the required information of every complete dataset of one instance of dronedata,  returns as list: [ 0: rootdir-path 1: annotation.shp-path, 2: annotation.tif-path , 3: droneimage.tif-path, 4: aoi.shp-path]"""
    #required following folderstructure: regionnamefolder -> instancenamefolder -> all files required + output-folder 
    #Annotaions files have to start with : Annotation AOi files: AOI Imagery: IMG 
    AllPaths =  {}
    completefolder = []
    namelist = []
    regionnames = [sub for sub in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, sub))]
    regionfolders = []
    for region in regionnames:
        regionfolders.append(os.path.join(root_dir,region))

    for instances in regionfolders:
        instancesfolder = [sub for sub in os.listdir(instances) if os.path.isdir(os.path.join(instances, sub))]
        for folder in instancesfolder:
            namelist.append(folder)
            completefolder.append(os.path.join(instances,folder))

    #find data
    for i, regionspath in enumerate(completefolder):
        Datalist = ["","","","",""]
        Datalist[0] = regionspath
        for file in os.listdir(regionspath):
            if file.startswith("AOI") and file.endswith(".shp"):
                Datalist[2] = file         
            if file.startswith("Annotation") and file.endswith(".shp"):
                Datalist[3] = file                
            if file.startswith("Annotation") and file.endswith(".tif"):
                Datalist[4] = file 

        for imagery in os.listdir(os.path.join(regionspath,ImgFileName)):
            if imagery.startswith("IMG") and imagery.endswith(".tif"):
                Datalist[1] = imagery

        for j, elements in enumerate(Datalist):
            print(elements)
            if Datalist[j] == "":
                print("Data is missing in: ", regionspath)
                print("provided data: ", Datalist)
                assert("Error")


        AllPaths[namelist[i]] = Datalist

    return AllPaths

def main(args):
    #Select Region to predict
    ImgFileName = args.img_dir
    model = EasyExperiment.load_from_checkpoint(args.checkpoint)
    model.eval()
    model.freeze()
    model.cuda()
    Paths = get_all_region_paths(args.root_dir,ImgFileName)
    tile_size = args.tile_size
    border = args.border

    for elements in Paths:
        #region = Paths[Regionname]
        region = Paths[elements]
        Regionname = elements
        outputpath = os.path.join(args.output_path, Regionname) + ".tif"
        aval_path = "" 
        #os.path.join(region[0], region[4]),

        print("AOI: ",os.path.join(region[0],region[2]))
        print("IMG: ", os.path.join(region[0],ImgFileName))
        print("Outputpath: ", outputpath)
        print("means: ", get_all_means_stds(Regionname)[0], get_all_means_stds(Regionname)[1], get_all_means_stds(Regionname)[2])
        print("stds: ", get_all_means_stds(Regionname)[3] ,get_all_means_stds(Regionname)[4], get_all_means_stds(Regionname)[5])

        test_set = AvalancheDatasetGrid(root_dir= os.path.join(region[0],ImgFileName),
                                        region_file= os.path.join(region[0],region[2]),
                                        dem_path=args.dem_path,
                                        aval_path=aval_path,
                                        tile_size=tile_size,
                                        overlap=border,
                                        bands=[1, 2, 3],
                                        means=[get_all_means_stds(Regionname)[0], get_all_means_stds(Regionname)[1], get_all_means_stds(Regionname)[2]],
                                        stds=[get_all_means_stds(Regionname)[3] ,get_all_means_stds(Regionname)[4], get_all_means_stds(Regionname)[5]],
                                        )

        test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4,
                                drop_last=False, pin_memory=True)





        pixel_w = test_set.pixel_w
        out_raster = create_raster(os.path.join(region[0],region[2]), outputpath, tile_size, pixel_w)
        out_band = out_raster.GetRasterBand(1)
        ulx, uly, _, _ = data_utils.get_raster_extent(out_raster)

        if aval_path != "":
            metrics = init_metrics()

        for sample in tqdm(iter(test_loader), desc='Predicting'):
            x = sample['input'].cuda()
            y_hat = model(x)
            y_hat = crop_to_center(y_hat, border)

            write_prediction(out_band, y_hat, sample['coords'], border, ulx, uly, pixel_w)

            if aval_path != "":
                y = sample['ground truth'].cuda()
                y = crop_to_center(y[:, [0], :, :], border)
                compute_metrics(metrics, y, y_hat)

    if aval_path != "":
        print('Finished. Computing metrics:')
        print_metrics(metrics)

def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

def file_path(string):
    if os.path.isfile(string):
        return string
    else:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), string)


if __name__ == "__main__":
    parser = ArgumentParser(description='Run avalanche prediction on satellite images')

    # Trainer args
    parser.add_argument('--img_dir', type=str, required = True, help='name of the IMG_dir')
    parser.add_argument('--root_dir', type=str, required = True, help='directory containing regions')
    parser.add_argument('--dem_path', type=str, required = True, default='', help='path to DEM if needed')
    parser.add_argument('--output_path', type=str, required = True, help='path to output file of predictions. Will be created or overwritten.')
    parser.add_argument('--checkpoint', type=str, required = True, help='model checkpoint to use')
    parser.add_argument('--tile_size', type=int, default=1024, help='Tile size to be used for predictions. Default: 1024')
    parser.add_argument('--border', type=int, default=650, help='Border to be disregarded for each sample in pixels. Default: 100')
    args = parser.parse_args()
    main(args)

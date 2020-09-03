import os
import geopandas as gpd
from torch.utils.data import Dataset, DataLoader
from osgeo import gdal, ogr
from utils import data_utils, viz_utils


class AvalancheDataset(Dataset):
    """
    SLF Avalanche Dataset. Samples are taken from avalanches within specified region

    :param root_dir: directory in which all data is located
    :param shape_file: shapefile name located in root_dir of the avalanches
    :param region_file: shapefile containing polygon specifying which area will be considered by the Dataset
    :param transform: transform to apply to data. Eg. rotation, toTensor, etc.
    :return pytorch dataset to be used with dataloader
    """

    def __init__(self, root_dir, shape_file, region_file, transform=None):
        self.transform = transform

        # open satellite images - all tiffs found in root directory
        all_tiffs = []
        for dirpath, _, filenames in os.walk(root_dir):
            for filename in [f for f in filenames if f.endswith(".tif") or f.endswith(".TIF")]:
                all_tiffs.append(os.path.join(dirpath, filename))
        self.vrt = gdal.BuildVRT('/tmp/myVRT.vrt', all_tiffs)

        geo_transform = self.vrt.GetGeoTransform()
        self.pixel_w = geo_transform[1]  # pixel width eg. 1 pixel => 1.5m

        # get x and y coordinates of upper left corner
        self.ulx, self.uly, _, _ = data_utils.get_raster_extent(self.vrt)

        # get avalanche shapes
        shape_path = os.path.join(root_dir, shape_file)
        self.avalanches = gpd.read_file(shape_path)

        # get sample points within region
        region = gpd.read_file(os.path.join(root_dir, region_file))
        self.avalanches = data_utils.get_avalanches_in_region(self.avalanches, region)

    def __len__(self):
        return len(self.avalanches)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        :param idx: index
        :return: [image, rasterised avalanches] as list
        """
        aval = self.avalanches.iloc[idx]

        offset = ((coord.x - self.ulx) / self.pixel_w, (self.uly - coord.y) / self.pixel_w)

        image = data_utils.get_all_bands_as_numpy(self.vrt, offset, self.res)
        shp_image = data_utils.get_numpy_from_shapefile(self.avalanches, self.vrt, offset, self.res)
        # image = image[:, :, 0:3]

        if self.transform:
            image = self.transform(image)
            shp_image = self.transform(shp_image)

        return [image, shp_image]


if __name__ == '__main__':
    # run test

    # data_folder = '/media/patrick/Seagate Expansion Drive/SLF_Avaldata/2019'
    data_folder = '/home/patrick/ecovision/data/2019'
    ava_file = 'avalanches0119_endversion.shp'
    # region_file = 'Region_Selection.shp'
    region_file = 'Multiple_regions.shp'

    my_dataset = AvalancheDataset(data_folder, ava_file, region_file)
    dataloader = DataLoader(my_dataset, batch_size=1, shuffle=True, num_workers=1)

    dataiter = iter(dataloader)
    batch = next(dataiter)

    batch = [elem.squeeze() for elem in batch]
    viz_utils.viz_sample(batch)
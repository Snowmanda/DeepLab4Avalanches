import os
from argparse import ArgumentParser
from pytorch_lightning import Trainer, seed_everything, Callback
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from experiments.easy_experiment import EasyExperiment
from experiments.inst_segm import InstSegmentation
from datasets.avalanche_dataset_points import AvalancheDatasetPoints
from datasets.avalanche_inst_dataset import AvalancheInstDataset
from datasets.davos_gt_dataset import DavosGtDataset
from torch.utils.data import DataLoader, ConcatDataset
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from utils.utils import str2bool, ba_collate_fn, inst_collate_fn
import matplotlib
matplotlib.use('Agg')  # dont open plots when training


def main(hparams):
    # Seed everything for reproducibility and more fair comparisons
    seed_everything(hparams.seed)

    # change some params when using Instance segmentation
    my_experiment = EasyExperiment
    my_dataset = AvalancheDatasetPoints
    my_collate_fn = ba_collate_fn
    print(hparams.model )
    if hparams.model == 'mask_rcnn':
        my_experiment = InstSegmentation
        my_dataset = AvalancheInstDataset
        my_collate_fn = inst_collate_fn

    # load model in different ways depending on whether training should be continued or a checkpoint is given
    if hparams.checkpoint:
        if hparams.resume_training:
            model = my_experiment(hparams)
            resume_ckpt = hparams.checkpoint
        else:
            model = my_experiment.load_from_checkpoint(hparams.checkpoint, hparams=hparams, strict=False)
            resume_ckpt = None
    else:
        model = my_experiment(hparams)
        resume_ckpt = None

    # Create the pytorch lightning trainer object to handle training along with tensorboard logging and checkpointing
    # Checkpoints are only saved when the soft_dice metric improves
    mylogger = TensorBoardLogger(hparams.log_dir, name=hparams.exp_name, default_hp_metric=False)
    mycheckpoint = ModelCheckpoint(monitor='f1/a_soft_dice', mode='max')
    trainer = Trainer.from_argparse_args(hparams, logger=mylogger, 
                                         resume_from_checkpoint=resume_ckpt,
                                         callbacks=[LearningRateMonitor('step'), mycheckpoint])

    # create the datasets needed. If the second dataset args are filled, use them as well and concatenate them into
    # one big dataset
    train_set = my_dataset(hparams.train_root_dir,
                           hparams.train_ava_file,
                           hparams.train_region_file,
                           dem_path=hparams.dem_dir,
                           tile_size=hparams.tile_size,
                           bands=hparams.bands,
                           certainty=hparams.aval_certainty,
                           batch_augm=hparams.batch_augm,
                           means=hparams.means,
                           stds=hparams.stds,
                           random=True,
                           hflip_p=hparams.hflip_p,
                           rand_rot=hparams.rand_rotation,
                           )
    if hparams.train_root_dir2:
        train_set2 = my_dataset(hparams.train_root_dir2,
                                hparams.train_ava_file2,
                                hparams.train_region_file2,
                                dem_path=hparams.dem_dir,
                                tile_size=hparams.tile_size,
                                bands=hparams.bands,
                                certainty=hparams.aval_certainty,
                                batch_augm=hparams.batch_augm,
                                means=hparams.means,
                                stds=hparams.stds,
                                random=True,
                                hflip_p=hparams.hflip_p,
                                rand_rot=hparams.rand_rotation,
                                )
        train_set = ConcatDataset([train_set, train_set2])

    val_set = my_dataset(hparams.val_root_dir,
                         hparams.val_ava_file,
                         hparams.val_region_file,
                         dem_path=hparams.dem_dir,
                         tile_size=2048,
                         bands=hparams.bands,
                         certainty=None,
                         batch_augm=0,
                         means=hparams.means,
                         stds=hparams.stds,
                         random=False,
                         hflip_p=0,
                         rand_rot=0,
                         )
    if hparams.val_root_dir2:
        val_set2 = my_dataset(hparams.val_root_dir2,
                              hparams.val_ava_file2,
                              hparams.val_region_file2,
                              dem_path=hparams.dem_dir,
                              tile_size=2048,
                              bands=hparams.bands,
                              certainty=None,
                              batch_augm=0,
                              means=hparams.means,
                              stds=hparams.stds,
                              random=False,
                              hflip_p=0,
                              rand_rot=0,
                              )
        val_set = ConcatDataset([val_set, val_set2])

    # ensure the batch size remains the same per gpu when using batch augmentation. Since the dataset already returns
    # multiple samples the dataloader does not need to load as many samples to make the same batch size
    loader_batch_size = hparams.batch_size // hparams.batch_augm if hparams.batch_augm > 0 else hparams.batch_size
    train_loader = DataLoader(train_set, batch_size=loader_batch_size, shuffle=True, num_workers=hparams.num_workers,
                              drop_last=True, pin_memory=True, collate_fn=my_collate_fn)
    val_loader = DataLoader(val_set, batch_size=hparams.batch_size, shuffle=False, num_workers=hparams.num_workers,
                            drop_last=False, pin_memory=True, collate_fn=my_collate_fn)

    # Run the training loop
    trainer.fit(model, train_loader, val_loader)

    # Test and compare on davos ground truth data when fininshed
    #test_set = DavosGtDataset(hparams.val_root_dir,
    #                          hparams.val_gt_file,
    #                          hparams.val_ava_file,
    #                          dem_path=hparams.dem_dir,
    #                          tile_size=512,
    #                          bands=hparams.bands,
    #                          means=hparams.means,
    #                          stds=hparams.stds,
    #                          )
    #if hparams.val_gt_file2:
    #    test_set2 = DavosGtDataset(hparams.val_root_dir2,
    #                               hparams.val_gt_file2,
    #                               hparams.val_ava_file2,
    #                               dem_path=hparams.dem_dir,
    #                               tile_size=512,
    #                               bands=hparams.bands,
    #                               means=hparams.means,
    #                               stds=hparams.stds,
    #                               )
    #    test_set = ConcatDataset([test_set, test_set2])

    #test_loader = DataLoader(test_set, batch_size=hparams.batch_size, shuffle=False, num_workers=hparams.num_workers,
    #                         drop_last=False, pin_memory=True)
    #trainer.test(test_dataloaders=test_loader)
#

if __name__ == "__main__":
    parser = ArgumentParser(description='train avalanche mapping network')

    # Trainer args
    parser.add_argument('--date', type=str, default='None', help='date when experiment was run')
    parser.add_argument('--time', type=str, default='None', help='time when experiment was run')
    parser.add_argument('--exp_name', type=str, default="default", help='experiment name')
    parser.add_argument('--seed', type=int, default=42, help='seed to init all random generators for reproducibility')
    parser.add_argument('--log_dir', type=str, default=os.getcwd(), help='directory to store logs and checkpoints')
    parser.add_argument('--checkpoint', type=str, default='', help='path to checkpoint if one is to be used')
    parser.add_argument('--resume_training', type=str2bool, default=False,
                        help='whether to resume training or only load model weights from checkpoint')

    # Dataset Args
    parser = AvalancheDatasetPoints.add_argparse_args(parser)

    # Dataset paths
    parser.add_argument('--train_root_dir', type=str, default='/home/patrick/ecovision/data/2018',
                        help='root directory of the training set')
    parser.add_argument('--train_ava_file', type=str, default='avalanches0118_endversion.shp',
                        help='File name of avalanche shapefile in root directory of training set')
    parser.add_argument('--train_region_file', type=str, default='Region_Selection.shp',
                        help='File name of shapefile in root directory defining training area')
    parser.add_argument('--val_root_dir', type=str, default='/home/patrick/ecovision/data/2018',
                        help='root directory of the validation set')
    parser.add_argument('--val_ava_file', type=str, default='avalanches0118_endversion.shp',
                        help='File name of avalanche shapefile in root directory of training set')
    parser.add_argument('--val_region_file', type=str, default='Region_Selection.shp',
                        help='File name of shapefile in root directory defining validation area')
    parser.add_argument('--dem_dir', type=str, default=None,
                        help='directory of the DEM within root_dir')
    #parser.add_argument('--val_gt_file', type=str, default='Methodenvergleich2018.shp',
    #                    help='File name of gt comparison data in davos')

    # use these when combining two datasets - 2018 and 2019
    parser.add_argument('--train_root_dir2', type=str, default='',
                        help='root directory of the training set 2')
    parser.add_argument('--train_ava_file2', type=str, default='',
                        help='File name of avalanche shapefile in root directory of training set 2')
    parser.add_argument('--train_region_file2', type=str, default='',
                        help='File name of shapefile in root directory defining training area 2')
    parser.add_argument('--val_root_dir2', type=str, default='',
                        help='root directory of the validation set 2')
    parser.add_argument('--val_ava_file2', type=str, default='',
                        help='File name of avalanche shapefile in root directory of training set 2')
    parser.add_argument('--val_region_file2', type=str, default='',
                        help='File name of shapefile in root directory defining validation area 2')
    #parser.add_argument('--val_gt_file2', type=str, default='',
    #                    help='File name of gt comparison data in davos 2')

    # Model specific args
    parser = EasyExperiment.add_model_specific_args(parser)

    parser = Trainer.add_argparse_args(parser)
    hparams = parser.parse_args()

    main(hparams)

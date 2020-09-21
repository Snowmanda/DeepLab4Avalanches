import torch
from torch.nn import L1Loss, MSELoss, BCELoss
from models.deep_lab_v4 import DeepLabv4
import pytorch_lightning as pl
from pytorch_lightning import TrainResult, EvalResult
from utils.losses import get_precision_recall_f1, recall_for_label
from utils import viz_utils, data_utils
from argparse import ArgumentParser


class EasyExperiment(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        self.model = DeepLabv4(in_channels=hparams.in_channels)
        self.bce_loss = BCELoss()
        self.l1 = L1Loss()
        self.mse = MSELoss()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.bce_loss(y_hat, data_utils.labels_to_mask(y))

        result = TrainResult(loss)
        result.log('train loss', loss, on_epoch=True, sync_dist=True)
        # Log random images
        if self.global_step % self.hparams.train_viz_interval == 0:
            image = viz_utils.viz_training(x, y, y_hat)
            self.logger.experiment.add_image("Training Sample", image, self.global_step)
        return result

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        pred = torch.round(y_hat) # rounds probability to 0 or 1

        bce_loss = self.bce_loss(y_hat, data_utils.labels_to_mask(y))
        precision, recall, f1 = get_precision_recall_f1(y, pred)
        recall1 = recall_for_label(y, pred, 1)
        recall2 = recall_for_label(y, pred, 2)
        recall3 = recall_for_label(y, pred, 3)


        # Logging metrics
        result = EvalResult(checkpoint_on=bce_loss)
        result.log('val bce loss', bce_loss, sync_dist=True)
        result.log('precision', precision, sync_dist=True, reduce_fx=nanmean)
        result.log('recall', recall, sync_dist=True, reduce_fx=nanmean)
        result.log('f1 Score', f1, sync_dist=True, reduce_fx=nanmean)
        result.log('recall exact', recall1, sync_dist=True, reduce_fx=nanmean)
        result.log('recall estimated', recall2, sync_dist=True, reduce_fx=nanmean)
        result.log('recall created', recall3, sync_dist=True, reduce_fx=nanmean)
        if batch_idx == self.hparams.val_viz_idx:
            image = viz_utils.viz_training(x, y, y_hat, pred)
            self.logger.experiment.add_image("Validation Sample", image, self.global_step)
        return result

    @staticmethod
    def add_model_specific_args(parent_parser):
        # allows adding model specific args via command line and logging them
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--lr', type=float, default=1e-3, help="learning rate of optimisation algorithm")
        parser.add_argument('--in_channels', type=int, default=4, help="no. of input channels to network")
        parser.add_argument('--train_viz_interval', type=int, default=100, help="image save interval during training")
        parser.add_argument('--val_viz_idx', type=int, default=0, help="batch index to be plotted during validation")
        return parser


def nanmean(x):
    """ Calculate mean ignoring nan values"""
    return x[~torch.isnan(x)].mean()

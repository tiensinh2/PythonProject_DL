import argparse
import datetime
import os
import torch
from torch import nn as nn
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
import sys

from torch.utils.tensorboard import SummaryWriter

from src.util.logconf import logging
from src.util.util import *
from src.dataset.dataset import LunaDataset
from src.dataset.model import LunaModel

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

METRICS_LABEL_NDX = 0
METRICS_PRED_NDX = 1
METRICS_LOSS_NDX = 2
METRICS_SIZE = 3

class LunaTrainingApp:
    def __init__(self, sys_argv = None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]
        parser = argparse.ArgumentParser()
        parser.add_argument('--num-workers',
                            help='number of worker processes',
                            type=int, default=0)
        parser.add_argument('--batch-size', type=int, default=16)
        parser.add_argument('--epochs', type=int, default= 3)
        parser.add_argument('--tb-prefix',
                            default='p2ch11',
                            help="Data prefix to use for Tensorboard run. Defaults to chapter.",
                            )

        parser.add_argument('comment',
                            help="Comment suffix for Tensorboard run.",
                            nargs='?',
                            default='dwlpt',
                            )
        self.cli_args = parser.parse_args(sys_argv)
        self.time_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.trn_writer = None
        self.val_writer = None
        self.totalTraining_samples = 0
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.model = self.initModel()
        self.optimizer = self.initOptimizer()
    def initModel(self):
        model = LunaModel()
        if self.use_cuda:
            log.info("Using CUDA; {} device".format(torch.cuda.device_count()))
            model = model.cuda()
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            model = model.to(self.device)
        return model
    def initOptimizer(self):
        return SGD(self.model.parameters(), lr = 0.01, momentum = 0.92)
    def initTrainDl(self):
        train_dts = LunaDataset(
            val_stride= 10,
            isValSet_bool = False,
        )
        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()
        train_dl = DataLoader(train_dts, batch_size = batch_size, num_workers = self.cli_args.num_workers, collate_fn=lambda batch: train_dts.collate_luna(batch),pin_memory = self.use_cuda)
        return train_dl
    def initValDl(self):
        val_ds = LunaDataset(
            val_stride=10,
            isValSet_bool=True,
        )

        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        val_dl = DataLoader(
            val_ds,
            batch_size=batch_size,
            num_workers=self.cli_args.num_workers,
            collate_fn=lambda batch: val_ds.collate_luna(batch),
            pin_memory=self.use_cuda,
        )
        return val_dl

    def initTensorBoardWriter(self):
        if self.trn_writer is None:
            log_dir = os.path.join("runs", self.time_str, self.cli_args.tb_prefix)
            self.trn_writer = SummaryWriter(log_dir=log_dir + '-trn_cls-' + self.cli_args.comment)
            self.val_writer = SummaryWriter(log_dir= log_dir + '-val_cls-' + self.cli_args.comment)
    def main(self):
        log.info("Starting training; {}, {}".format(type(self).__name__, self.cli_args))
        train_dl = self.initTrainDl()
        val_dl = self.initValDl()
        for epoch in range(1, self.cli_args.epochs+1):
            log.info("Epoch {} of {}, {}/{} batches of size {}*{}".format(
                epoch,
                self.cli_args.epochs,
                len(train_dl),
                len(val_dl),
                self.cli_args.batch_size,
                (torch.cuda.device_count() if self.use_cuda else 1),
            ))
            trainMetrics = self.doTraining(epoch, train_dl)
            ## log metric
            self.logMetric(epoch, 'trn', trainMetrics)
            valMetrics = self.doValidation(epoch, val_dl)
            ##log metric
            self.logMetric(epoch, 'val', valMetrics)
        if hasattr(self, 'trn_writer'):
            self.trn_writer.close()
            self.val_writer.close()

    def doTraining(self, epoch_ndx, train_dl):
        self.model.train()
        trainMetrics = torch.zeros(METRICS_SIZE,
                                   len(train_dl.dataset),
                                   device = self.device)
        batch_iter = enumerateWithEstimate(train_dl,
                                           "E{} Training".format(epoch_ndx),
                                           start_ndx= train_dl.num_workers)


        for batch_ndx, batch_tup in enumerate(batch_iter):
            logging.debug(f"Batch index: {batch_ndx}")
            logging.debug(f"Batch type: {type(batch_tup)}")
            logging.debug(f"Batch length: {len(batch_tup)}")
            logging.debug(f"Batch content types: {[type(x) for x in batch_tup]}")
            logging.debug(f"Batch shapes: {[x.shape if isinstance(x, torch.Tensor) else len(x) for x in batch_tup]}")
            self.optimizer.zero_grad()
            loss = self.computeBatchLoss(batch_ndx,
                                         batch_tup,
                                         train_dl.batch_size,
                                         trainMetrics)
            loss.backward()
            self.optimizer.step()

            if epoch_ndx == 1 and batch_ndx == 0:
                with torch.no_grad():
                    self.trn_writer.add_graph(self.model, batch_tup[0], verbose=True)
        self.totalTraining_samples += len(train_dl.dataset)
        return trainMetrics.to('cpu')
    def doValidation(self, epoch_ndx, val_dl):
        self.model.eval()
        valMetrics = torch.zeros(METRICS_SIZE,
                                 len(val_dl.dataset),
                                 device = self.device)
        batch_iter = enumerateWithEstimate(val_dl,
                                           "E{} Validation".format(epoch_ndx),
                                           start_ndx= val_dl.num_workers)
        for batch_ndx, batch_tup in enumerate(batch_iter):
            with torch.no_grad():
                loss_val = self.computeBatchLoss(batch_ndx,
                                                 batch_tup,
                                                 val_dl.batch_size,
                                                 valMetrics)
        self.totalTraining_samples += len(val_dl.dataset)
        return valMetrics.to('cpu')
    def computeBatchLoss(self, batch_ndx, batch_tup, batch_size, metrics_g):
        inputs, labels, _series_list, _center_list = batch_tup
        input_g = inputs.to(self.device, non_blocking=True)
        labels_g = labels.to(self.device, non_blocking=True)

        logits_g, probability_g = self.model(input_g)
        loss_func = nn.CrossEntropyLoss(reduction='none')
        loss_g = loss_func(logits_g, labels_g[:, 1])
        start_ndx = batch_ndx * batch_size
        end_ndx = start_ndx + labels.size(0)
        metrics_g[METRICS_LABEL_NDX, start_ndx:end_ndx] = \
            labels_g[:, 1].detach()
        metrics_g[METRICS_PRED_NDX, start_ndx:end_ndx] = \
            probability_g[:, 1].detach()
        metrics_g[METRICS_LOSS_NDX, start_ndx:end_ndx] = \
            loss_g.detach()

        return loss_g.mean()
    def logMetric(self, epoch_ndx, mode_str, metrics_t, classificationThreshold = 0.5):
        self.initTensorBoardWriter()
        log.info("E{} {}".format(epoch_ndx, type(self).__name__))
        negLabel_mask = metrics_t[METRICS_LABEL_NDX] <= classificationThreshold
        negPred_mask = metrics_t[METRICS_PRED_NDX] <= classificationThreshold
        posLabel_mask = ~negLabel_mask
        posPred_mask = ~negPred_mask
        neg_count = int(negLabel_mask.sum())
        pos_count = int(posLabel_mask.sum())
        neg_correct = int((negPred_mask & negLabel_mask).sum())
        pos_correct = int((posPred_mask & posLabel_mask).sum())
        metrics_dict = {}
        metrics_dict['loss/all'] = \
            metrics_t[METRICS_LOSS_NDX].mean()
        metrics_dict['loss/neg'] = \
            metrics_t[METRICS_LOSS_NDX, negLabel_mask].mean()
        metrics_dict['loss/pos'] = \
            metrics_t[METRICS_LOSS_NDX, posLabel_mask].mean()

        metrics_dict['correct/all'] = (pos_correct + neg_correct) \
                                      / np.float32(metrics_t.shape[1]) * 100
        metrics_dict['correct/neg'] = neg_correct / np.float32(neg_count) * 100
        metrics_dict['correct/pos'] = pos_correct / np.float32(pos_count) * 100

        log.info(
            ("E{} {:8} {loss/all:.4f} loss, "
             + "{correct/all:-5.1f}% correct, "
             ).format(
                epoch_ndx,
                mode_str,
                **metrics_dict,
            )
        )
        log.info(
            ("E{} {:8} {loss/neg:.4f} loss, "
             + "{correct/neg:-5.1f}% correct ({neg_correct:} of {neg_count:})"
             ).format(
                epoch_ndx,
                mode_str + '_neg',
                neg_correct=neg_correct,
                neg_count=neg_count,
                **metrics_dict,
            )
        )
        log.info(
            ("E{} {:8} {loss/pos:.4f} loss, "
             + "{correct/pos:-5.1f}% correct ({pos_correct:} of {pos_count:})"
             ).format(
                epoch_ndx,
                mode_str + '_pos',
                pos_correct=pos_correct,
                pos_count=pos_count,
                **metrics_dict,
            )
        )
        writer = getattr(self, mode_str + '_writer')
        for key, value in metrics_dict.items():
            writer.add_scalar(key, value, self.totalTraining_samples)
        writer.add_pr_curve(
            'pr',
            metrics_t[METRICS_PRED_NDX],
            metrics_t[METRICS_LABEL_NDX],
            self.totalTraining_samples
        )
        bins = [x/50.0 for x in range(51)]
        negHist_mask = negLabel_mask & (metrics_t[METRICS_PRED_NDX] > 0.01)
        posHist_mask = posLabel_mask & (metrics_t[METRICS_PRED_NDX] < 0.99)

        if negHist_mask.any():
            writer.add_histogram(
                'is_neg',
                metrics_t[METRICS_PRED_NDX, negHist_mask],
                self.totalTraining_samples,
                bins=bins,
            )
        if posHist_mask.any():
            writer.add_histogram(
                'is_pos',
                metrics_t[METRICS_PRED_NDX, posHist_mask],
                self.totalTraining_samples,
                bins=bins,
            )
if __name__ == '__main__':
    LunaTrainingApp().main()








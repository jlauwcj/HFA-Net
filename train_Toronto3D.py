# Common
import os
import logging
import warnings
import argparse
import numpy as np
from tqdm import tqdm
# torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# my module
from dataset.Toronto3D_trainset import Toronto3D
from utils.config import ConfigToronto3D as cfg
from utils.metric import compute_acc, IoUCalculator
from network.loss_func import compute_loss
import datetime
from functools import partialmethod
from network.focal_loss import FocalLoss
import shutil
# reproductability
import torch.backends.cudnn as cudnn
import random

# torch.manual_seed(43)
manual_seed = 3407
# if manual_seed is not None:
random.seed(manual_seed)
np.random.seed(manual_seed)
torch.manual_seed(manual_seed)
torch.cuda.manual_seed(manual_seed)
torch.cuda.manual_seed_all(manual_seed)
cudnn.benchmark = False
cudnn.deterministic = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False

# disable tqdm progressing bar
tqdm.__init__ = partialmethod(tqdm.__init__, disable=False)
warnings.filterwarnings("ignore")
# /media/wcj/9C922B9C922B7A42/zl/RandLA-pytorch/log/our0.778/checkpoint.tar
parser = argparse.ArgumentParser()
parser.add_argument('--backbone', type=str, default='randla', choices=['randla', 'baflac', 'baaf'])
parser.add_argument('--checkpoint_path', default=None, help='Model checkpoint path [default: None]')
parser.add_argument('--log_dir', default='default', help='dir prefix to save model checkpoint [default: default]')
parser.add_argument('--max_epoch', type=int, default=100, help='Epoch to run [default: 100]')
parser.add_argument('--batch_size', type=int, default=2, help='Batch Size during training [default: 6]')
parser.add_argument('--val_batch_size', type=int, default=8,
                    help='Validation Batch Size during training [default: 30]')
parser.add_argument('--num_workers', type=int, default=10, help='Number of workers [default: 10]')
parser.add_argument('--focal', type=bool, default=True, help='If use focal loss or not[default: True]')
parser.add_argument('--syn', type=int, default=0, help='whether use synkitti mixing or pure kitti')
parser.add_argument('--focal_gamma', type=int, default=2, help='gamma for focal loss[default: 2]')
parser.add_argument('--ignore', type=str, default='major', help='mode to ignore default or major [choice: major]')
FLAGS = parser.parse_args()

def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def get_logger(filename):
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    LOGGING_FORMAT = '%(asctime)s %(levelname)s: %(message)s'

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(logging.Formatter(LOGGING_FORMAT))
    logger.addHandler(fh)

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(LOGGING_FORMAT))
    logger.addHandler(handler)
    return logger


class Trainer:
    def __init__(self):
        # Init Logging
        FLAGS.log_dir = "/media/wcj/9C922B9C922B7A42/zl/RandLA-pytorch/log/" + FLAGS.log_dir + '_' + datetime.datetime.now().strftime(
            "%Y-%-m-%d-%H-%M")
        if not os.path.exists(FLAGS.log_dir):
            os.makedirs(FLAGS.log_dir)
        self.log_dir = FLAGS.log_dir
        log_fname = os.path.join(FLAGS.log_dir, 'log_train.txt')
        LOGGING_FORMAT = '%(asctime)s %(levelname)s: %(message)s'
        DATE_FORMAT = '%Y%m%d %H:%M:%S'
        logging.basicConfig(level=logging.DEBUG, format=LOGGING_FORMAT, datefmt=DATE_FORMAT, filename=log_fname)
        self.logger = get_logger(log_fname)
        # store current config
        shutil.copy('/media/wcj/9C922B9C922B7A42/zl/RandLA-pytorch/train_Semantic3d.py', FLAGS.log_dir)
        shutil.copy('/media/wcj/9C922B9C922B7A42/zl/RandLA-pytorch/utils/config.py', FLAGS.log_dir)
        shutil.copy('/media/wcj/9C922B9C922B7A42/zl/RandLA-pytorch/dataset/semantic3d_trainset.py', FLAGS.log_dir)
        shutil.copy('/media/wcj/9C922B9C922B7A42/zl/RandLA-pytorch/network/RandLANet3dis.py', FLAGS.log_dir)

        # tensorboard writer
        self.tf_writer = SummaryWriter(self.log_dir)

        if FLAGS.ignore == 'major':
            ignore_labels = [0]
        else:
            ignore_labels = None

        train_dataset = Toronto3D('train', ignore_labels=ignore_labels)
        val_dataset = Toronto3D('val', ignore_labels=ignore_labels)

        # Network & Optimizer
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if FLAGS.backbone == 'baflac':
            from network.BAF_LAC import BAF_LAC
            self.logger.info("Use Baseline: BAF-LAC")
            self.net = BAF_LAC(cfg)
            self.net.to(device)
            collate_fn = train_dataset.collate_fn_baf_lac

        elif FLAGS.backbone == 'randla':
            from network.RandLANet3dis import Network
            self.logger.info("Use Baseline: Rand-LA")
            self.net = Network(cfg)
            self.net.to(device)
            collate_fn = train_dataset.collate_fn

        elif FLAGS.backbone == 'baaf':
            from network.BAAF import Network
            self.logger.info("Use Baseline: BAAF")
            self.net = Network(cfg)
            self.net.to(device)
            collate_fn = train_dataset.collate_fn

        else:
            raise TypeError("1~5~!! can can need !!!")

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=FLAGS.batch_size,
            shuffle=True,
            num_workers=FLAGS.num_workers,
            worker_init_fn=None,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=True,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=FLAGS.val_batch_size,
            shuffle=True,
            num_workers=FLAGS.num_workers,
            worker_init_fn=None,
            collate_fn=collate_fn,
            pin_memory=True
        )

        # Load the Adam optimizer
        # self.optimizer = optim.Adam(self.net.parameters(), lr=cfg.learning_rate)

        # use the adamw
        self.optimizer = optim.AdamW(self.net.parameters(),lr=cfg.learning_rate,weight_decay=0.001)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, 0.95)

        # Load module
        self.start_epoch = 0
        CHECKPOINT_PATH = FLAGS.checkpoint_path
        if CHECKPOINT_PATH is not None and os.path.isfile(CHECKPOINT_PATH):
            checkpoint = torch.load(CHECKPOINT_PATH)
            print("Loading pre_trained models:", CHECKPOINT_PATH)
            self.net.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.start_epoch = checkpoint['epoch']
        self.logger.info("Training Config: Batch size: %d, max_epoch: %d, start epoch: %d" % (
        FLAGS.batch_size, FLAGS.max_epoch, self.start_epoch))

        # Loss Function
        class_weights = torch.from_numpy(train_dataset.get_class_weight()).float().to(device)
        if not FLAGS.focal:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='none')
        else:
            # use focal loss
            self.criterion = FocalLoss(class_weights, FLAGS.focal_gamma)
            # self.criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='none')

        # log configurations
        self.logger.info(FLAGS)

        # Multiple GPU Training
        if torch.cuda.device_count() > 1:
            self.logger.info("Let's use %d GPUs!" % (torch.cuda.device_count()))
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            self.net = nn.DataParallel(self.net)

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        '''
        # 验证加载器参数是否相同，鉴定为真
        for key, value in self.train_dataset.__dict__.items():
            print(key,":", value)
        for key, value in self.val_dataset.__dict__.items():
            print(key,":", value)
        exit()
        '''


    def train_one_epoch(self):
        self.net.train()  # set model to training mode
        '''
        # 验证模型初始化是否相同，鉴定为真
        for name, param in self.net.named_parameters():
            if param.requires_grad:
                print(name, param.data)
        exit()
        '''

        iou_calc = IoUCalculator(cfg)
        tqdm_loader = tqdm(self.train_loader, total=len(self.train_loader))
        mLoss = 0
        mAcc = 0
        for batch_idx, batch_data in enumerate(tqdm_loader):
            # batch_data即为经过collect_fn后的inputs
            # {'xyz':[..],'color':[..],[..],...'nei..':[..],...,'features':[..],'label':[0,5..]}
            # print(batch_data)
            for key in batch_data:
                if type(batch_data[key]) is list:
                    for i in range(cfg.num_layers):
                        batch_data[key][i] = batch_data[key][i].cuda(non_blocking=True)
                else:
                    batch_data[key] = batch_data[key].cuda(non_blocking=True)

            self.optimizer.zero_grad()
            # Forward pass
            torch.cuda.synchronize()
            #print(torch.cuda.synchronize())
            #print("111")
            # 用于确保前面的所有CUDA操作都已完成。它可以帮助确保在进行下一步之前，之前的CUDA计算已经完成。
            end_points = self.net(batch_data)
            # 将批次数据 batch_data 输入到模型中进行前向传播。end_points 是模型输出的结果。
            loss, end_points = compute_loss(end_points, self.train_dataset, self.criterion)

            loss.backward()
            self.optimizer.step()

            # for evaluataion the training process
            acc, end_points = compute_acc(end_points)
            mAcc += acc.item()
            mLoss += loss.item()
            iou_calc.add_data(end_points)

        self.tf_writer.add_scalar("Loss/Train", mLoss / batch_idx, self.cur_epoch)
        self.tf_writer.add_scalar("Acc/Train", mAcc / batch_idx, self.cur_epoch)
        mean_iou, _ = iou_calc.compute_iou()
        freqweight_iou = iou_calc.compute_freqweighted_iou()
        self.tf_writer.add_scalar("mIoU/Train", mean_iou, self.cur_epoch)
        self.tf_writer.add_scalar("freqweight_iou/Train", freqweight_iou, self.cur_epoch)
        # print("Epoch: {}, Loss: {}, Acc: {}, mIoU: {}, freqweighted_IoU: {}".format(self.cur_epoch, mLoss/batch_idx, mAcc/batch_idx, mean_iou, freqweight_iou))
        self.logger.info(
            "Epoch: {}, Loss: {}, Acc: {}, mIoU: {}, freqweighted_IoU: {}".format(self.cur_epoch, mLoss / batch_idx,
                                                                                  mAcc / batch_idx, mean_iou,
                                                                                  freqweight_iou))
        self.scheduler.step()

    def train(self):
        highest_miou = 0
        highest_fwIoU = 0
        for epoch in range(self.start_epoch, FLAGS.max_epoch):
            self.cur_epoch = epoch
            self.logger.info('**** EPOCH %03d ****' % (epoch))

            self.train_one_epoch()
            self.logger.info('**** EVAL EPOCH %03d ****' % (epoch))
            mean_iou, fw_IoU = self.validate()

            # Save best checkpoint
            if mean_iou > highest_miou:
                # print("saving best models with mIoU: ", mean_iou)
                self.logger.info("saving best models with mIoU:{} ".format(mean_iou))
                highest_miou = mean_iou
                checkpoint_file = os.path.join(self.log_dir, 'checkpoint.tar')
                self.save_checkpoint(checkpoint_file)
            '''
            # save model with best fwmIoU
            if fw_IoU > highest_fwIoU:
                # print("saving best models with fwIoU: ", fw_IoU)
                self.logger.info("saving best models with fwIoU:{} ".format(fw_IoU))
                highest_fwIoU = fw_IoU
                checkpoint_file = os.path.join(self.log_dir, 'fw_checkpoint.tar')
                self.save_checkpoint(checkpoint_file)
            '''

            checkpoint_file = os.path.join(self.log_dir, 'latest_checkpoint.tar')
            self.save_checkpoint(checkpoint_file)

    def validate(self):
        self.net.eval()  # set model to eval mode (for bn and dp)
        iou_calc = IoUCalculator(cfg)
        mloss = 0
        macc = 0
        div = 1e-5
        tqdm_loader = tqdm(self.val_loader, total=len(self.val_loader))
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(tqdm_loader):
                # print(batch_data)
                div += 1
                for key in batch_data:
                    if type(batch_data[key]) is list:
                        for i in range(cfg.num_layers):
                            batch_data[key][i] = batch_data[key][i].cuda(non_blocking=True)
                    else:
                        batch_data[key] = batch_data[key].cuda(non_blocking=True)

                # Forward pass
                torch.cuda.synchronize()
                end_points = self.net(batch_data)

                loss, end_points = compute_loss(end_points, self.train_dataset, self.criterion)
                mloss += loss.item()
                acc, end_points = compute_acc(end_points)
                macc += acc.item()
                iou_calc.add_data(end_points)
        self.tf_writer.add_scalar("Loss/Valid", mloss / div, self.cur_epoch)
        self.tf_writer.add_scalar("Acc/Valid", macc / div, self.cur_epoch)
        mean_iou, iou_list = iou_calc.compute_iou()
        freqweight_iou = iou_calc.compute_freqweighted_iou()
        self.tf_writer.add_scalar("mIoU/Valid", mean_iou, self.cur_epoch)
        self.tf_writer.add_scalar("freqweight_iou/Valid", freqweight_iou, self.cur_epoch)
        self.logger.info('mean IoU:{:.1f}, freqweight_iou: {:.1f}'.format(mean_iou * 100, freqweight_iou * 100))
        s = 'IoU:'
        for iou_tmp in iou_list:
            s += '{:5.2f} '.format(100 * iou_tmp)
        self.logger.info(s)
        # print("Valid Epoch: {}, Loss: {}, Acc: {}, mIoU: {}, fwIoU: {}".format(self.cur_epoch, mloss/batch_idx, macc/batch_idx, mean_iou, freqweight_iou))
        self.logger.info(
            "Valid Epoch: {}, Loss: {}, Acc: {}, mIoU: {}, fwIoU: {}".format(self.cur_epoch, mloss / batch_idx,
                                                                             macc / batch_idx, mean_iou,
                                                                             freqweight_iou))
        return mean_iou, freqweight_iou

    def save_checkpoint(self, fname):
        save_dict = {
            'epoch': self.cur_epoch + 1,  # after training one epoch, the start_epoch should be epoch+1
            'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict()
        }
        # with nn.DataParallel() the net is added as a submodule of DataParallel
        try:
            save_dict['model_state_dict'] = self.net.module.state_dict()
        except AttributeError:
            save_dict['model_state_dict'] = self.net.state_dict()
        torch.save(save_dict, fname)


def main():
    trainer = Trainer()
    trainer.train()


if __name__ == '__main__':
    main()

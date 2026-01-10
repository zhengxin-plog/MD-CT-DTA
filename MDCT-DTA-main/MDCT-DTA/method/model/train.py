import os
import math
import numpy as np
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import logging
from datetime import datetime
from torch_geometric.data import DataLoader
from metrics import get_cindex
from dataset import *
from model1 import MDCTDTA
from utils import *

class SimpleLogger:
    def __init__(self, params):
        self.dataset = params.get("dataset", "default")
        self.save_dir = params.get("save_dir", "save")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model_dir = os.path.join(self.save_dir, f"{self.dataset}_{timestamp}")
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        # 配置 logging 格式
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(os.path.join(self.model_dir, "train.log")),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger()

    def info(self, msg):
        self.logger.info(msg)

    def get_model_dir(self):
        return self.model_dir


def val(model, criterion, dataloader, device):
    model.eval()
    running_loss = AverageMeter()

    for data in dataloader:
        data = data.to(device)

        with torch.no_grad():
            pred = model(data)
            loss = criterion(pred.view(-1), data.y.view(-1))
            label = data.y
            running_loss.update(loss.item(), label.size(0))

    epoch_loss = running_loss.get_average()
    running_loss.reset()
    model.train()
    return epoch_loss


def main():
    torch.autograd.set_detect_anomaly(True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="davis", help='davis, kiba, metz or bindingdb')  #required=True
    parser.add_argument('--save_model', default="save model", help='whether save model or not')
    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=512, help='batch_size')
    args = parser.parse_args()

    params = dict(
        data_root="data",
        save_dir="save",
        dataset=args.dataset,
        save_model=args.save_model,
        lr=args.lr,
        batch_size=args.batch_size
    )

    logger = SimpleLogger(params)
    logger.info(f"Using Script: {__file__}")

    DATASET = params.get("dataset")
    save_model = params.get("save_model")
    data_root = params.get("data_root")
    fpath = os.path.join(data_root, DATASET)

    # 检查数据集路径是否存在
    if not os.path.exists(fpath):
        logger.info(f"Error: Dataset path {fpath} not found!")
        return

    train_set = GNNDataset(fpath, train=True)
    test_set = GNNDataset(fpath, train=False)


    logger.info(f"Train set size: {len(train_set)}")
    logger.info(f"Test set size: {len(test_set)}")

    train_loader = DataLoader(train_set, batch_size=params['batch_size'], shuffle=True, num_workers=0)  # Windows建议设为0
    test_loader = DataLoader(test_set, batch_size=params['batch_size'], shuffle=False, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    model = MDCTDTA(3, 25 + 1, embedding_size=128, filter_num=32, out_dim=1).to(device)

    epochs = 500
    steps_per_epoch = 50
    num_iter = math.ceil((epochs * steps_per_epoch) / len(train_loader))
    break_flag = False

    optimizer = optim.Adam(model.parameters(), lr=params['lr'])
    criterion = nn.MSELoss()

    global_step = 0
    global_epoch = 0
    early_stop_epoch = 500

    running_loss = AverageMeter()
    running_cindex = AverageMeter()
    running_best_mse = BestMeter("min")

    model.train()

    for i in range(num_iter):
        if break_flag:
            break

        for data in train_loader:
            global_step += 1
            data = data.to(device)
            pred = model(data)

            loss = criterion(pred.view(-1), data.y.view(-1))

            try:
                cindex = get_cindex(data.y.detach().cpu().numpy().reshape(-1), pred.detach().cpu().numpy().reshape(-1))
            except:
                cindex = 0.5

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss.update(loss.item(), data.y.size(0))
            running_cindex.update(cindex, data.y.size(0))

            if global_step % steps_per_epoch == 0:
                global_epoch += 1

                epoch_loss = running_loss.get_average()
                epoch_cindex = running_cindex.get_average()
                running_loss.reset()
                running_cindex.reset()

                test_loss = val(model, criterion, test_loader, device)

                msg = "num_iter-%d, epoch-%d, loss-%.4f, cindex-%.4f, test_loss-%.4f" % (i, global_epoch, epoch_loss,
                                                                                         epoch_cindex, test_loss)
                logger.info(msg)

                if test_loss < running_best_mse.get_best():
                    running_best_mse.update(test_loss)
                    if save_model:
                        save_model_dict(model, logger.get_model_dir(), "best_model")
                else:
                    count = running_best_mse.counter()
                    if count > early_stop_epoch:
                        logger.info(f"early stop in epoch {global_epoch}")
                        break_flag = True
                        break


if __name__ == "__main__":
    main()
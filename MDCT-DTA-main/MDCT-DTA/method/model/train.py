# %%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
import math
import numpy as np
import torch.optim as optim
import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
import torch.nn.functional as F
import argparse

from metrics import get_cindex
from dataset import *
from ML_DTI import MGraphDTA
from utils import *
from log.train_logger import TrainLogger

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
    parser = argparse.ArgumentParser()
    #2e-4 5e-3 test_loss:0.2340, test_cindex:0.8931, test_r2:0.6889 'num_iter-6123, epoch-409, loss-0.0953, cindex-0.9478, test_loss-0.2328.pt'
    #2e-4 2e-4 test_loss:0.2068, test_cindex:0.9033, test_r2:0.7320, 20230607_150335_davis/model/'num_iter-6123, epoch-1313, loss-0.0481, cindex-0.9676, test_loss-0.2064.pt'
    # Add argument
    parser.add_argument('--dataset', required=True, help='davis or kiba')
    parser.add_argument('--save_model', action='store_true', help='whether save model or not')
    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=512, help='batch_size')
    args = parser.parse_args()
    # python train.py --dataset davis --lr 8e-3 --batch_size 256 testloss 0.4
    params = dict(
        data_root="data",
        save_dir="save",
        dataset=args.dataset,
        save_model=args.save_model,
        lr=args.lr,
        batch_size=args.batch_size
    )

    logger = TrainLogger(params)
    logger.info(__file__)

    DATASET = params.get("dataset")
    save_model = params.get("save_model")
    data_root = params.get("data_root")
    fpath = os.path.join(data_root, DATASET)

    train_set = GNNDataset(fpath, train=True)
    test_set = GNNDataset(fpath, train=False)

    print(len(train_set))
    print(len(test_set))

    train_loader = DataLoader(train_set, batch_size=512, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_set, batch_size=512, shuffle=False, num_workers=8)

    device = torch.device('cuda:2')
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MGraphDTA(3, 25 + 1, embedding_size=128, filter_num=32, out_dim=1).to(device)

    # epochs = 3000
    epochs = 6000;
    steps_per_epoch = 50
    num_iter = math.ceil((epochs * steps_per_epoch) / len(train_loader))
    break_flag = False

    optimizer = optim.Adam(model.parameters(), lr=3e-4)
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
            # print("-----------------------train data:",data,"------------------------")
            pred = model(data)

            loss = criterion(pred.view(-1), data.y.view(-1))
            cindex = get_cindex(data.y.detach().cpu().numpy().reshape(-1), pred.detach().cpu().numpy().reshape(-1))

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

                msg = "num_iter-%d, epoch-%d, loss-%.4f, cindex-%.4f, test_loss-%.4f" % (num_iter, global_epoch, epoch_loss, epoch_cindex, test_loss)
                logger.info(msg)

                if test_loss < running_best_mse.get_best():
                    running_best_mse.update(test_loss)
                    if save_model:
                        save_model_dict(model, logger.get_model_dir(), msg)
                else:
                    count = running_best_mse.counter()
                    if count > early_stop_epoch:
                        logger.info(f"early stop in epoch {global_epoch}")
                        break_flag = True
                        break

if __name__ == "__main__":
    main()

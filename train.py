r""" Hypercorrelation Squeeze training (validation) code """
import argparse

import torch.optim as optim
import torch.nn as nn
import torch

from docs.HSNet.Model.HSNet import HypercorrSqueezeNetwork
from docs.HSNet.Common.logger import Logger, AverageMeter
from docs.HSNet.Common.Evaluator import Evaluator
from docs.HSNet.Common import Utils as utils
from docs.HSNet.DataLoader.FSSDataset import FSSDataset
from docs.HSNet.Common.Visualizer import Visualizer
from tqdm import tqdm
import datetime


import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

def train(epoch, model, dataloader, optimizer, training):
    r""" Train HSNet """

    # Force randomness during training / freeze randomness during testing
    utils.fix_randseed(None) if training else utils.fix_randseed(0)
    model.module.train_mode() if training else model.module.eval()
    average_meter = AverageMeter(dataloader.dataset)

    for idx, batch in enumerate(dataloader):

        # 1. Hypercorrelation Squeeze Networks forward pass
        batch = utils.to_cuda(batch)
        logit_mask = model(batch['query_img'], batch['support_imgs'].squeeze(1), batch['support_masks'].squeeze(1))
        pred_mask = logit_mask.argmax(dim=1)
        
        # 2. Compute loss & update model parameters
        loss = model.module.compute_objective(logit_mask, batch['query_mask'])
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 3. Evaluate prediction
        area_inter, area_union = Evaluator.classify_prediction(pred_mask, batch)
        average_meter.update(area_inter, area_union, batch['class_id'], loss.detach().clone())
        average_meter.write_process(idx, len(dataloader), epoch, write_batch_idx=50)

        # 4. visualise validation data
        if not training:
            if Visualizer.visualize:
                Visualizer.visualize_prediction_batch(batch['support_imgs'], batch['support_masks'],
                                                  batch['query_img'], batch['query_mask'],
                                                  pred_mask, idx, iou_b=area_inter / area_union)

    # Write evaluation results
    average_meter.write_result('Training' if training else 'Validation', epoch)
    avg_loss = utils.u_mean(average_meter.loss_buf)
    miou, fb_iou = average_meter.compute_iou()

    return avg_loss, miou, fb_iou


if __name__ == '__main__':

    # Arguments parsing
    parser = argparse.ArgumentParser(description='Hypercorrelation Squeeze Pytorch Implementation')
    parser.add_argument('--datapath', type=str, default='DataSet/')
    parser.add_argument('--benchmark', type=str, default='pascal', choices=['pascal', 'coco', 'fss'])
    parser.add_argument('--logpath', type=str, default= datetime.datetime.now().__format__('_%m%d_%H%M%S'))
    parser.add_argument('--bsz', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--niter', type=int, default=2000)
    parser.add_argument('--nworker', type=int, default=8)
    parser.add_argument('--fold', type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument('--backbone', type=str, default='resnet101', choices=['vgg16', 'resnet50', 'resnet101'])
    parser.add_argument('--visualize', type=bool, default=True)

    args = parser.parse_args()
    Logger.initialize(args, training=True)

    # Model initialization
    model = HypercorrSqueezeNetwork(args.backbone, False)
    Logger.log_params(model)

    # Device setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Logger.info('# available GPUs: %d' % torch.cuda.device_count())
    model = nn.DataParallel(model)
    model.to(device)

    # Helper classes (for training) initialization
    optimizer = optim.Adam([{"params": model.parameters(), "lr": args.lr}])
    Evaluator.initialize()

    # Dataset initialization
    FSSDataset.initialize(img_size=400, datapath=args.datapath, use_original_imgsize=False)
    dataloader_trn = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'trn')
    dataloader_val = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'val')

    # Vis init
    Visualizer.initialize(args.visualize, args.logpath)

    # Train HSNet
    best_val_miou = float('-inf')
    best_val_loss = float('inf')
    for epoch in tqdm(range(args.niter)):

        trn_loss, trn_miou, trn_fb_iou = train(epoch, model, dataloader_trn, optimizer, training=True)
        with torch.no_grad():
            val_loss, val_miou, val_fb_iou = train(epoch, model, dataloader_val, optimizer, training=False)

        # Save the best model
        if val_miou > best_val_miou:
            best_val_miou = val_miou
            torch.save(model.state_dict(), os.path.join(args.logpath, 'logs/models/model_val_'+str(val_miou.item())[0:4]+'.pt'))
            print('Model saved @%d w/ val. mIoU: %5.2f.\n' % (epoch, val_miou))

        Logger.tbd_writer.add_scalars('data/loss', {'trn_loss': trn_loss, 'val_loss': val_loss}, epoch)
        Logger.tbd_writer.add_scalars('data/miou', {'trn_miou': trn_miou, 'val_miou': val_miou}, epoch)
        Logger.tbd_writer.add_scalars('data/fb_iou', {'trn_fb_iou': trn_fb_iou, 'val_fb_iou': val_fb_iou}, epoch)
        Logger.tbd_writer.flush()
    Logger.tbd_writer.close()
    Logger.info('==================== Finished Training ====================')
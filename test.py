r""" Hypercorrelation Squeeze Testing code """
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
import pandas as pd

def test(model, dataloader, nshot):
    r""" Test HSNet """

    # Freeze randomness during testing for reproducibility
    utils.fix_randseed(0)
    average_meter = AverageMeter(dataloader.dataset)

    # collect info about run
    eval = []

    for idx, batch in enumerate(dataloader):

        # 1. Hypercorrelation Squeeze Networks forward pass
        batch = utils.to_cuda(batch)
        pred_mask = model.module.predict_mask_nshot(batch, nshot=nshot)

        assert pred_mask.size() == batch['query_mask'].size()

        # 2. Evaluate prediction
        area_inter, area_union, eval_dict = Evaluator.classify_prediction(pred_mask.clone(), batch)
        average_meter.update(area_inter, area_union, batch['class_id'], loss=None)
        average_meter.write_process(idx, len(dataloader), epoch=-1, write_batch_idx=1)

        # Visualize predictions
        if Visualizer.visualize:
            Visualizer.visualize_prediction_batch(batch['support_imgs'], batch['support_masks'],
                                                  batch['query_img'], batch['query_mask'],
                                                  pred_mask, idx, iou_b=area_inter / area_union)

        eval.append(eval_dict)

    # Write evaluation results
    average_meter.write_result('Test', 0)
    miou, fb_iou = average_meter.compute_iou()

    return miou, fb_iou, eval


if __name__ == '__main__':

    # Arguments parsing
    parser = argparse.ArgumentParser(description='Hypercorrelation Squeeze Pytorch Implementation')
    parser.add_argument('--datapath', type=str, default='DataSet/')

    parser.add_argument('--benchmark', type=str, default='pascal', choices=['pascal', 'coco', 'fss'])
    parser.add_argument('--logpath', type=str, default= "TEST" + datetime.datetime.now().__format__('_%m%d_%H%M%S'))

    parser.add_argument('--bsz', type=int, default=1)
    parser.add_argument('--nworker', type=int, default=0)
    parser.add_argument('--load', type=str, default='best_model.pt')
    parser.add_argument('--fold', type=int, default=2, choices=[0, 1, 2, 3])
    parser.add_argument('--nshot', type=int, default=1)
    parser.add_argument('--backbone', type=str, default='resnet101', choices=['vgg16', 'resnet50', 'resnet101'])
    parser.add_argument('--visualize', type=bool, default=True)
    parser.add_argument('--use_original_imgsize', action='store_true')
    parser.add_argument('--neg_inst_rate', type=bool, default=True)

    args = parser.parse_args()
    Logger.initialize(args)


    # Model initialization
    model = HypercorrSqueezeNetwork(args.backbone, args.use_original_imgsize)
    model.eval()
    Logger.log_params(model)

    # Device setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Logger.info('# available GPUs: %d' % torch.cuda.device_count())
    model = nn.DataParallel(model)
    model.to(device)

    # Load trained model
    if args.load == '': raise Exception('Pretrained model not specified.')
    model.load_state_dict(torch.load("logs/models/" + args.load))

    # Helper classes (for testing) initialization
    Evaluator.initialize()
    Visualizer.initialize(args.visualize, args.logpath)

    # Dataset initialization
    FSSDataset.initialize(img_size=400, datapath=args.datapath, use_original_imgsize=args.use_original_imgsize)
    dataloader_test = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'test', args.nshot)

    # Test HSNet
    with torch.no_grad():
        test_miou, test_fb_iou, eval = test(model, dataloader_test, args.nshot)

    # eval_list saved for analysis
    df = pd.DataFrame.from_dict(eval)
    df.to_csv("logs/" + args.logpath + ".log/eval.csv")


    Logger.info('Fold %d mIoU: %5.2f \t FB-IoU: %5.2f' % (args.fold, test_miou.item(), test_fb_iou.item()))
    Logger.info('==================== Finished Testing ====================')
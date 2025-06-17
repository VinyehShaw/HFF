import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import argparse
import time
import os
import numpy as np
import random
from tqdm import tqdm
import wandb
import nibabel as nib

from config.train_test_config.train_test_config import print_val_loss, print_val_eval, save_val_best_3d_m
from config.warmup_config.warmup import GradualWarmupScheduler
from loss.loss_function import segmentation_loss
from model.HFF import HFFNet
from loader.dataload3d import get_loaders
from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)


def init_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def make_label_mapping(dataset_name, class_type):
    if dataset_name in ('brats19','brats20','msdbts'):
        raw = [0,1,2,4]
    else:
        raw = [0,1,2,3]
    if class_type == 'et':
        pos = raw[-1]
        mapping = {l:(1 if l==pos else 0) for l in raw}
    elif class_type == 'tc':
        p1, p2 = 1, raw[-1]
        mapping = {l:(1 if l in (p1,p2) else 0) for l in raw}
    elif class_type == 'wt':
        p2, p3 = 2, raw[-1]
        mapping = {l:(1 if l in (1,p2,p3) else 0) for l in raw}
    else:
        mapping = { old:new for new, old in enumerate(raw) }
    return mapping


def mask_to_class_indices(mask, mapping):
    out = torch.zeros_like(mask, dtype=torch.long)
    for old, new in mapping.items():
        out[mask == old] = new
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HFF-Net 3D Inference')
    parser.add_argument('--test_list', type=str,  help='Path to text file listing test volumes',default='./bratsxxx/x-val.txt')
    parser.add_argument('--checkpoint', type=str,  help='Path to trained model checkpoint (.pth)',default=' yourpath/your_trained_hff-net.pth')
    parser.add_argument('--dataset_name', choices=['brats19','brats20','brats23men','msdbts'], default='brats19')
    parser.add_argument('--class_type', choices=['et','tc','wt','all'], default='et')
    parser.add_argument('--selected_modal', nargs='+', default=[
        'flair_L','t1_L','t1ce_L','t2_L',
        'flair_H1','flair_H2','flair_H3','flair_H4',
        't1_H1','t1_H2','t1_H3','t1_H4',
        't1ce_H1','t1ce_H2','t1ce_H3','t1ce_H4',
        't2_H1','t2_H2','t2_H3','t2_H4'], help='Modalities')
    # Brats 23
    # parser.add_argument('--selected_modal', nargs='+',
    #                     default=['t2w_L', 't1n_L', 't1c_L', 't2f_L', 't2f_H1', 't2f_H2', 't2f_H3', 't2f_H4',
    #                              't1n_H1', 't1n_H2', 't1n_H3', 't1n_H4', 't1c_H1',
    #                              't1c_H2', 't1c_H3', 't1c_H4', 't2w_H1', 't2w_H2', 't2w_H3', 't2w_H4'])
    
    
    parser.add_argument('-b','--batch_size', type=int, default=1)
    parser.add_argument('-l','--loss', type=str, default='dice')
    parser.add_argument('--loss2', type=str, default='ff')
    args = parser.parse_args()
    
    init_seeds(42)
    os.makedirs(args.output_dir, exist_ok=True)

    mapping = make_label_mapping(args.dataset_name, args.class_type)
    classnum = 4 if args.class_type=='all' else 2
    criterion = segmentation_loss(args.loss, False, cn=classnum).cuda()
    # load model
    model = HFFNet(4,16,classnum).cuda()
    state_dict = torch.load(args.checkpoint, map_location='cuda')
    model.load_state_dict(state_dict)
    model.eval()

   
    # loader: using same list for train and val since get_loaders expects both
    data_files = dict(train=args.test_list, val=args.test_list)
    loaders = get_loaders(data_files, args.selected_modal, args.batch_size, num_workers=4)
    val_loader = loaders['val']
    num_batches = len(val_loader)
    val_loss_sup_1 = 0.0
    val_loss_sup_2 = 0.0

    # inference
    all_scores, all_masks = [], []
    with torch.no_grad():
        score_list_val_1 = None
        score_list_val_2 = None
        mask_list_val = None
        for i, data in enumerate(tqdm(val_loader, desc='Inference')):
            # unpack modalities and mask
            # data is list of 21 tensors: 20 modalities + mask
            low_freq_inputs = []
            high_freq_inputs = []
            for j in range(20):
                tensor = data[j].unsqueeze(1).cuda()
                if j in [0,1,2,3]:
                    low_freq_inputs.append(tensor)
                else:
                    high_freq_inputs.append(tensor)
            low = torch.cat(low_freq_inputs, dim=1)
            high = torch.cat(high_freq_inputs, dim=1)
            mask_val = mask_to_class_indices(data[20], mapping).long().cuda()

            outputs_val_1, outputs_val_2, side1, side2 = model(low, high)
            outputs_val_1_cpu = outputs_val_1.detach().cpu()
            outputs_val_2_cpu = outputs_val_2.detach().cpu()
            mask_cpu = mask_val.detach().cpu()

            if i == 0:
                score_list_val_1 = outputs_val_1_cpu
                score_list_val_2 = outputs_val_2_cpu
                mask_list_val = mask_cpu
            else:
                score_list_val_1 = torch.cat((score_list_val_1, outputs_val_1_cpu), dim=0)
                score_list_val_2 = torch.cat((score_list_val_2, outputs_val_2_cpu), dim=0)
                mask_list_val = torch.cat((mask_list_val, mask_cpu), dim=0)

            loss1 = criterion(outputs_val_1_cpu, mask_cpu)
            loss2 = criterion(outputs_val_2_cpu, mask_cpu)
            val_loss_sup_1 += loss1.item()
            val_loss_sup_2 += loss2.item()

        # summarize
        print_val_loss(val_loss_sup_1, val_loss_sup_2, {'val': num_batches}, 63, 0)
        print_val_eval(classnum, score_list_val_1, score_list_val_2, mask_list_val, 31)

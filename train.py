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


from config.train_test_config.train_test_config import print_train_loss, print_val_loss, \
    print_val_eval, print_best,save_val_best_3d_m
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
    """
    Returns a dict old_label->new_label:
    - For 'brats19','brats20','msdbts', raw labels = [0,1,2,4]
    - For 'brats23men',             raw labels = [0,1,2,3]
    - class_type in {'et','tc','wt','all'} controls how we collapse.
    """
    if dataset_name in ('brats19','brats20','msdbts'):
        raw = [0,1,2,4]
    else:  # 'brats23men'
        raw = [0,1,2,3]

    if class_type == 'et':
        #  enhancing = {1, last}
        pos = raw[-1]
        mapping = {l:(1 if l==pos else 0) for l in raw}
    elif class_type == 'tc':
        # core = {1, last}
        p1, p2 = 1, raw[-1]
        mapping = {l:(1 if l in (p1,p2) else 0) for l in raw}
    elif class_type == 'wt':
        # whole = {1,2,last}
        p2, p3 = 2, raw[-1]
        mapping = {l:(1 if l in (1,p2,p3) else 0) for l in raw}
    else:  # all → 连续映射到 [0..len(raw)-1]
        mapping = { old:new for new, old in enumerate(raw) }
    # print(f" raw labels = {raw}, class_type = {class_type}, mapping = {mapping}")
    return mapping

def mask_to_class_indices(mask, mapping):
    """
    Recode `mask` according to `mapping: old_label->new_label`. D-enhancing tumor (ET — label 4), the peritumoral edema (ED — label 2), and the necrotic and non-enhancing tumor core (NCR/NET — label 1),
    (ET, label 4), regions of the tumor core (TC, labels 1 and 4), and the whole tumor region (WT, labels 1,2 and 4)
    """
    out = torch.zeros_like(mask, dtype=torch.long)
    for old, new in mapping.items():
        out[mask == old] = new
    return out

def compute_fisher_information(model, dataloader, criterion):
    model.eval()
    fisher_information = {}
    for name, param in model.named_parameters():
        if name =='input_ed.conv.weight':
            fisher_information[name] = torch.zeros_like(param)

    for i, data in enumerate(dataloader):
        low_freq_inputs = []
        high_freq_inputs = []
        for j in range(20):  # to fit model input
            input_tensor = data[j].unsqueeze(dim=1).type(torch.cuda.FloatTensor)
            if j in [0, 1, 2, 3]:
                low_freq_inputs.append(input_tensor)
            else:
                high_freq_inputs.append(input_tensor)
        low_freq_inputs = torch.cat(low_freq_inputs, dim=1)
        high_freq_inputs = torch.cat(high_freq_inputs, dim=1)
        target = mask_to_class_indices(data[20], label_mapping).long().cuda()
        outputs_train_1, outputs_train_2,side1,side2= model(low_freq_inputs, high_freq_inputs)
        loss_train_sup1 = criterion(outputs_train_1, target)
        loss_train_sup2 = criterion(outputs_train_2, target)
        total_loss = loss_train_sup1 + loss_train_sup2
        model.zero_grad()
        total_loss.backward()
        for name, param in model.named_parameters():
            if name in fisher_information:
                fisher_information[name] += param.grad ** 2

    # Average over the number of samples
    for name in fisher_information:
        fisher_information[name] /= len(dataloader)

    return fisher_information


def important_weights_with_fisher(model, fisher_info,  std_multiplier=1):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name == 'input_ed.conv.weight':
                importance = fisher_info[name]
                # Calculate mean and standard deviation
                mean_importance = torch.mean(importance, dim=[2, 3, 4], keepdim=True)
                std_importance = torch.std(importance, dim=[2, 3, 4], keepdim=True)
                # Set threshold
                threshold = mean_importance + std_multiplier * std_importance
                important_weights = importance > threshold
        return important_weights



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_list', type=str, default='./brats19/2-train.txt')
    parser.add_argument('--val_list', type=str, default='./brats19/2-val.txt')
    parser.add_argument('--path_trained_models', default='./result/checkpoints/hff')
    parser.add_argument(
    '--dataset_name',
    choices=['brats19','brats20','brats23men','msdbts'],
    default='brats19',
    help="Which BRATS/MSD dataset to use"
    )
    parser.add_argument(
        '--class_type',
        choices=['et','tc','wt','all'],
        default='all',
        help="Which segmentation target: ET (enhancing), TC (core), WT (whole), or ALL (keep all labels)"
    )
    # Brats 19/20/msd
    parser.add_argument('--selected_modal', nargs='+',
                        default=['flair_L','t1_L','t1ce_L','t2_L','flair_H1', 'flair_H2','flair_H3','flair_H4', 't1_H1', 't1_H2','t1_H3','t1_H4', 't1ce_H1',
                                 't1ce_H2','t1ce_H3','t1ce_H4', 't2_H1', 't2_H2','t2_H3','t2_H4'])
    # Brats 23
    # parser.add_argument('--selected_modal', nargs='+',
    #                     default=['t2w_L', 't1n_L', 't1c_L', 't2f_L', 't2f_H1', 't2f_H2', 't2f_H3', 't2f_H4',
    #                              't1n_H1', 't1n_H2', 't1n_H3', 't1n_H4', 't1c_H1',
    #                              't1c_H2', 't1c_H3', 't1c_H4', 't2w_H1', 't2w_H2', 't2w_H3', 't2w_H4'])
  

    parser.add_argument('--input1', default='L')
    parser.add_argument('--input2', default='H')
    parser.add_argument('--sup_mark', default='100', help='100')
    parser.add_argument('-b', '--batch_size', default=1, type=int)
    parser.add_argument('-e', '--num_epochs', default=450, type=int)
    parser.add_argument('-s', '--step_size', default=50, type=int)
    parser.add_argument('-l', '--lr', default=0.3, type=float)
    parser.add_argument('-g', '--gamma', default=0.55, type=float)
    parser.add_argument('-u', '--unsup_weight', default=15, type=float)
    parser.add_argument('--loss', default='dice', type=str)
    parser.add_argument('--loss2', default='ff', type=str)
    parser.add_argument('-w', '--warm_up_duration', default=40)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--wd', default=-5, type=float, help='weight decay pow')
    parser.add_argument('--queue_length', default=48, type=int)
    parser.add_argument('--samples_per_volume_train', default=4, type=int)
    parser.add_argument('--samples_per_volume_val', default=8, type=int) 
    parser.add_argument('-i', '--display_iter', default=1, type=int)
    parser.add_argument('-n', '--network', default='hff', type=str)
    # 
    

    args = parser.parse_args()

    wandb.init(project=str('learning rate=' + str(
        args.lr) + 'epochs=' + str(args.num_epochs) + '-step_size=' + str(args.step_size) + '-gamma=' + str(args.gamma) + 'weight between braches=' + str(args.unsup_weight) + 'warmup epochs=' + str(args.warm_up_duration) + str(args.dataset_name) + '_'+str(args.class_type) ))


    wandb.config.update(args)

    
    seed = 42
    init_seeds(seed)
    if args.class_type == 'all':
    # all retains all original labels (brats19/20/msd=[0,1,2,4], brats23men=[0,1,2,3]), 
    # in mapping, new_label=max(raw), so the category is the length of raw
        classnum = 4  
    else:
    # et/tc/wt are all binary classification
        classnum = 2
    dataset_name = args.dataset_name
    label_mapping = make_label_mapping(args.dataset_name, args.class_type)

    print_num = 63
    print_num_minus = print_num - 2
    print_num_half = int(print_num / 2 - 1)

    path_trained_models = args.path_trained_models + '/' + str(os.path.split(args.dataset_name)[1]+ '/' + str(args.class_type) )
    if not os.path.exists(path_trained_models):
        os.makedirs(path_trained_models)
    path_trained_models = path_trained_models + '/' + str(args.network) + '-l=' + str(args.lr) + '-e=' + str(
        args.num_epochs) + '-s=' + str(args.step_size) + '-g=' + str(args.gamma) + '-b=' + str(
        args.batch_size) + '-cw=' + str(args.unsup_weight) + '-w=' + str(args.warm_up_duration) + '-' + str(
        args.sup_mark) + str(args.input1) + '-' + str(args.input2)
    if not os.path.exists(path_trained_models):
        os.makedirs(path_trained_models)


    
    data_files = dict(train=args.train_list, val=args.val_list)
    loaders = get_loaders(data_files, args.selected_modal, args.batch_size, num_workers=8)
    loaders = {x: loaders[x] for x in ('train', 'val')}


    num_batches = {'train_sup': len(loaders['train']), 'val': len(loaders['val'])}

    # Model
    model = HFFNet(4,16, classnum)
    model = model.cuda()
    # model = DistributedDataParallel(model, device_ids=[args.local_rank])

    # Training Strategy
    criterion = segmentation_loss(args.loss, False,cn=classnum).cuda()
    FFcriterion = segmentation_loss(args.loss2, True).cuda()

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay= 5.3 * 10 ** args.wd)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1.0, total_epoch=args.warm_up_duration,
                                              after_scheduler=exp_lr_scheduler)

    # Train & Val
    since = time.time()
    count_iter = 0
    important_weights = None

    best_model = model
    best_result = 'Result1'
    best_val_eval_list = [0 for i in range(1)]

    for epoch in range(args.num_epochs):

        count_iter += 1
        if (count_iter - 1) % args.display_iter == 0:
            begin_time = time.time()



        # dataloaders['train'].sampler.set_epoch(epoch)
        model.train()

        train_loss_sup_1 = 0.0
        train_loss_sup_2 = 0.0
        train_loss_unsup = 0.0
        train_loss_reg=0.0

        train_loss_side1=0.0
        train_loss_side2=0.0
        train_loss_side3=0.0
        train_loss_side4=0.0

        train_loss = 0.0
        val_loss_sup_1 = 0.0
        val_loss_sup_2 = 0.0

        unsup_weight = args.unsup_weight * (epoch + 1) / args.num_epochs
        # reg_weight=0.00001* (epoch + 1) / args.num_epochs
        reg_weight = 0.000005
        # dist.barrier()

        if epoch == args.warm_up_duration:
        # if epoch == 20:
            fisher_info = compute_fisher_information(model, loaders['train'], criterion)
            important_weights = important_weights_with_fisher(model, fisher_info, std_multiplier=1)
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name == 'input_ed.conv.weight':
                        param[important_weights] = model.laplacian_target[important_weights]


        for i, data in enumerate(tqdm(loaders['train'])):
     

            low_freq_inputs = []
            high_freq_inputs = []

            for j in range(20):  # Sort by --selected_modal parameter
            
                input_tensor = data[j].unsqueeze(dim=1).type(torch.cuda.FloatTensor)

                # Sort by --selected_modal parameter
                if j in [0, 1, 2, 3]:
                    low_freq_inputs.append(input_tensor)
                else:
                    high_freq_inputs.append(input_tensor)
            low_freq_inputs = torch.cat(low_freq_inputs, dim=1)
            high_freq_inputs = torch.cat(high_freq_inputs, dim=1)
            mask_train = mask_to_class_indices(data[20],label_mapping).long().cuda()

            optimizer.zero_grad()
            outputs_train_1, outputs_train_2,side1,side2 = model(low_freq_inputs, high_freq_inputs)
            torch.cuda.empty_cache()
    
         
           

            loss_train_sup1 = criterion(outputs_train_1, mask_train)
            loss_train_sup2 = criterion(outputs_train_2, mask_train)
            loss_train_side1 = criterion(side1,mask_train)
            loss_train_side2 = criterion(side2,mask_train)
        

            reg_loss = torch.sum((model.input_ed.conv.weight - model.laplacian_target) ** 2)*reg_weight
           
            loss_train_unsup = FFcriterion(outputs_train_1,outputs_train_2)
            loss_train_unsup = loss_train_unsup * unsup_weight

            loss_train = loss_train_sup1 + loss_train_sup2 + loss_train_unsup+loss_train_side1+loss_train_side2+reg_loss
   

            loss_train.backward()


            if important_weights is not None:
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if name == "input_ed.conv.weight":
                            param.grad[important_weights] = 0


            optimizer.step()



            train_loss_sup_1 += loss_train_sup1.item()
            train_loss_sup_2 += loss_train_sup2.item()
            train_loss_reg +=reg_loss.item()
           
            train_loss_side1+=loss_train_side1.item()
            train_loss_side2+=loss_train_side2.item()
        
            train_loss_unsup += loss_train_unsup.item()
            train_loss += loss_train.item()

        wandb.log({"train_loss_sup1": train_loss_sup_1/ num_batches['train_sup'],
                   "train_loss_sup2": train_loss_sup_2/ num_batches['train_sup'],
                   "train_loss": train_loss/ num_batches['train_sup'],
                   "train_loss_side1": train_loss_side1/ num_batches['train_sup'],
                   "train_loss_side2": train_loss_side2/ num_batches['train_sup'],
                   # "train_loss_side3": train_loss_side3/ num_batches['train_sup'],
                   # "train_loss_side4": train_loss_side4/ num_batches['train_sup'],
                   "train_loss_unsup":  train_loss_unsup/ num_batches['train_sup'],
                   "reg_loss": train_loss_reg/ num_batches['train_sup'],

                   "epoch": epoch})

        scheduler_warmup.step()
        torch.cuda.empty_cache()

        if count_iter % args.display_iter == 0:

            print('=' * print_num)
            print('| Epoch {}/{}'.format(epoch + 1, args.num_epochs).ljust(print_num_minus, ' '), '|')
            train_epoch_loss_sup_1, train_epoch_loss_sup_2, train_epoch_loss_cps,reg1, train_epoch_loss = print_train_loss(
                train_loss_sup_1, train_loss_sup_2, train_loss_unsup,train_loss_reg, train_loss, num_batches, print_num,
                print_num_half)

            torch.cuda.empty_cache()

            with torch.no_grad():
                model.eval()

                for i, data in enumerate(loaders['val']):

      
                    low_freq_inputs = []
                    high_freq_inputs = []

                    for j in range(20):  # Sort by --selected_modal parameter
                        input_tensor = data[j].unsqueeze(dim=1).type(torch.cuda.FloatTensor)

                        # Sort by --selected_modal parameter
                        if j in [0, 1, 2, 3]:
                            low_freq_inputs.append(input_tensor)
                        else:
                            high_freq_inputs.append(input_tensor)
                    low_freq_inputs = torch.cat(low_freq_inputs, dim=1)
                    high_freq_inputs = torch.cat(high_freq_inputs, dim=1)
                    mask_val = mask_to_class_indices(data[20],label_mapping).long().cuda()

                    optimizer.zero_grad()
                    outputs_val_1, outputs_val_2,side1,side2 = model(low_freq_inputs, high_freq_inputs)
                    torch.cuda.empty_cache()
                    outputs_val_1 = outputs_val_1.detach().cpu()
                    outputs_val_2 = outputs_val_2.detach().cpu()
                    mask_val = mask_val.detach().cpu()
                    if i == 0:
                        score_list_val_1 = outputs_val_1
                        score_list_val_2 = outputs_val_2
                        mask_list_val = mask_val
                    else:
                        score_list_val_1 = torch.cat((score_list_val_1, outputs_val_1), dim=0)
                        score_list_val_2 = torch.cat((score_list_val_2, outputs_val_2), dim=0)
                        mask_list_val = torch.cat((mask_list_val, mask_val), dim=0)

                    loss_val_sup_1 = criterion(outputs_val_1, mask_val)
                    loss_val_sup_2 = criterion(outputs_val_2, mask_val)
                    val_loss_sup_1 += loss_val_sup_1.item()
                    val_loss_sup_2 += loss_val_sup_2.item()

                wandb.log({"val_loss_sup1": val_loss_sup_1 / num_batches['val'],
                           "val_loss_sup2": val_loss_sup_2 / num_batches['val'],
                           })



                torch.cuda.empty_cache()


                val_epoch_loss_sup_1, val_epoch_loss_sup_2 = print_val_loss(val_loss_sup_1, val_loss_sup_2, num_batches,
                                                                            print_num, print_num_half)
                val_eval_list_1, val_eval_list_2, val_m_dc_1, val_m_dc_2 = print_val_eval(classnum,
                                                                                          score_list_val_1,
                                                                                          score_list_val_2,
                                                                                          mask_list_val, print_num_half)
                # Modify the indicators that need to be displayed in wandb log by yourself
                # wandb.log({
                #             "Val ET Dice 1": val_eval_list_1[0],
                #             "Val ET HD95 1": val_eval_list_1[1],
                #             "Val TC Dice 1": val_eval_list_1[2],
                #             "Val TC HD95 1": val_eval_list_1[3],
                #             "Val WT Dice 1": val_eval_list_1[4],
                #             "Val WT HD95 1": val_eval_list_1[5],
                #             "Val ET Dice 2": val_eval_list_2[0],
                #             "Val ET HD95 2": val_eval_list_2[1],
                #             "Val TC Dice 2": val_eval_list_2[2],
                #             "Val TC HD95 2": val_eval_list_2[3],
                #             "Val WT Dice 2": val_eval_list_2[4],
                #             "Val WT HD95 2": val_eval_list_2[5]
                #         })
                wandb.log({
                        "Val WT Dice 1": val_eval_list_1[0],
                        "Val WT HD95 1": val_eval_list_1[1],
                        # "Val TC Dice 1": val_eval_list_1[2],
                        # "Val TC HD95 1": val_eval_list_1[3],
                        # "Val WT Dice 1": val_eval_list_1[4],
                        # "Val WT HD95 1": val_eval_list_1[5],
                        "Val WT Dice 2": val_eval_list_2[0],
                        "Val WT HD95 2": val_eval_list_2[1],
                        # "Val TC Dice 2": val_eval_list_2[2],
                        # "Val TC HD95 2": val_eval_list_2[3],
                        # "Val WT Dice 2": val_eval_list_2[4],
                        # "Val WT HD95 2": val_eval_list_2[5]
                    })
                
                best_val_eval_list, best_model, best_result = save_val_best_3d_m(classnum, best_model,
                                                                               best_val_eval_list, best_result, model,
                                                                               model, score_list_val_1,
                                                                               score_list_val_2, mask_list_val,
                                                                               val_eval_list_1, val_eval_list_2,
                                                                               path_trained_models,)

                torch.cuda.empty_cache()

             
                print('-' * print_num)
                print('| Epoch Time: {:.4f}s'.format((time.time() - begin_time) / args.display_iter).ljust(
                    print_num_minus, ' '), '|')
            torch.cuda.empty_cache()
        torch.cuda.empty_cache()

    time_elapsed = time.time() - since
    m, s = divmod(time_elapsed, 60)
    h, m = divmod(m, 60)

    print('=' * print_num)
    print('| Training Completed In {:.0f}h {:.0f}mins {:.0f}s'.format(h, m, s).ljust(print_num_minus, ' '), '|')
    print('-' * print_num)
    print_best(classnum, best_val_eval_list, best_model, best_result, path_trained_models, print_num_minus)
    print('=' * print_num)
    wandb.finish()




import numpy as np
from config.eval_config.eval import evaluate, evaluate_multi,evaluate_multi_binary, evaluate_groupwise, evaluate_groupwise_binary
import torch
import os
from PIL import Image
import torchio as tio

def print_train_loss_sup(train_loss, num_batches, print_num, print_num_minus):
    train_epoch_loss = train_loss / num_batches['train_sup']
    print('-' * print_num)
    print('| Train Loss: {:.4f}'.format(train_epoch_loss).ljust(print_num_minus, ' '), '|')
    print('-' * print_num)
    return train_epoch_loss

def print_train_loss_MT(train_loss_sup_1, train_loss_cps, train_loss, num_batches, print_num, print_num_half, print_num_minus):
    train_epoch_loss_sup1 = train_loss_sup_1 / num_batches['train_sup']
    train_epoch_loss_cps = train_loss_cps / num_batches['train_sup']
    train_epoch_loss = train_loss / num_batches['train_sup']
    print('-' * print_num)
    print('| Train  Sup  Loss: {:.4f}'.format(train_epoch_loss_sup1).ljust(print_num_half, ' '), '| Train Unsup Loss: {:.4f}'.format(train_epoch_loss_cps).ljust(print_num_half, ' '), '|')
    print('| Train Total Loss: {:.4f}'.format(train_epoch_loss).ljust(print_num_minus, ' '), '|')
    print('-' * print_num)
    return train_epoch_loss_sup1, train_epoch_loss_cps, train_epoch_loss

def print_train_loss_ConResNet(train_loss_seg, train_loss_res, train_loss, num_batches, print_num, print_num_half, print_num_minus):
    train_epoch_loss_seg = train_loss_seg / num_batches['train_sup']
    train_epoch_loss_res = train_loss_res / num_batches['train_sup']
    train_epoch_loss = train_loss / num_batches['train_sup']
    print('-' * print_num)
    print('| Train  Seg  Loss: {:.4f}'.format(train_epoch_loss_seg).ljust(print_num_half, ' '), '| Train Res Loss: {:.4f}'.format(train_epoch_loss_res).ljust(print_num_half, ' '), '|')
    print('| Train Total Loss: {:.4f}'.format(train_epoch_loss).ljust(print_num_minus, ' '), '|')
    print('-' * print_num)
    return train_epoch_loss_seg, train_epoch_loss_res, train_epoch_loss


def print_train_loss_EM(train_loss_sup_1, train_loss_cps, train_loss, num_batches, print_num, print_num_minus):
    train_epoch_loss_sup1 = train_loss_sup_1 / num_batches['train_sup']
    train_epoch_loss_cps = train_loss_cps / num_batches['train_sup']
    train_epoch_loss = train_loss / num_batches['train_sup']
    print('-' * print_num)
    print('| Train  Sup  Loss: {:.4f}'.format(train_epoch_loss_sup1).ljust(print_num_minus, ' '), '|')
    print('| Train Unsup Loss: {:.4f}'.format(train_epoch_loss_cps).ljust(print_num_minus, ' '), '|')
    print('| Train Total Loss: {:.4f}'.format(train_epoch_loss).ljust(print_num_minus, ' '), '|')
    print('-' * print_num)
    return train_epoch_loss_sup1, train_epoch_loss_cps, train_epoch_loss


def print_train_loss(train_loss_sup_1, train_loss_sup_2, train_loss_cps,reg1, train_loss, num_batches, print_num, print_num_half):
    train_epoch_loss_sup1 = train_loss_sup_1 / num_batches['train_sup']
    train_epoch_loss_sup2 = train_loss_sup_2 / num_batches['train_sup']
    train_epoch_loss_cps = train_loss_cps / num_batches['train_sup']
    reg1 = reg1 / num_batches['train_sup']
    # reg2 = reg2 / num_batches['train_sup']
    train_epoch_loss = train_loss / num_batches['train_sup']
    print('-' * print_num)
    print('| Train Sup Loss LF: {:.4f}'.format(train_epoch_loss_sup1).ljust(print_num_half, ' '), '| Train SUP Loss HF: {:.4f}'.format(train_epoch_loss_sup2).ljust(print_num_half, ' '), '|')
    print('| Train Unsup Loss: {:.4f}'.format(train_epoch_loss_cps).ljust(print_num_half, ' '), '| Train Total Loss: {:.4f}'.format(train_epoch_loss).ljust(print_num_half, ' '), '|')
    print('-' * print_num)
    return train_epoch_loss_sup1, train_epoch_loss_sup2, train_epoch_loss_cps,reg1, train_epoch_loss

def print_val_loss_sup(val_loss, num_batches, print_num, print_num_minus):
    val_epoch_loss = val_loss / num_batches['val']
    print('-' * print_num)
    print('| Val Loss: {:.4f}'.format(val_epoch_loss).ljust(print_num_minus, ' '), '|')
    print('-' * print_num)
    return val_epoch_loss

def print_val_loss(val_loss_sup_1, val_loss_sup_2, num_batches, print_num, print_num_half):
    val_epoch_loss_sup1 = val_loss_sup_1 / num_batches['val']
    val_epoch_loss_sup2 = val_loss_sup_2 / num_batches['val']
    print('-' * print_num)
    print('| Val Sup Loss LF: {:.4f}'.format(val_epoch_loss_sup1).ljust(print_num_half, ' '), '| Val Sup Loss HF: {:.4f}'.format(val_epoch_loss_sup2).ljust(print_num_half, ' '), '|')
    print('-' * print_num)
    return val_epoch_loss_sup1, val_epoch_loss_sup2

def print_val_loss_ConResNet(val_loss_seg, val_loss_res, num_batches, print_num, print_num_half):
    val_epoch_loss_seg = val_loss_seg / num_batches['val']
    val_epoch_loss_res = val_loss_res / num_batches['val']
    print('-' * print_num)
    print('| Val Seg Loss: {:.4f}'.format(val_epoch_loss_seg).ljust(print_num_half, ' '), '| Val Res Loss: {:.4f}'.format(val_epoch_loss_res).ljust(print_num_half, ' '), '|')
    print('-' * print_num)
    return val_epoch_loss_seg, val_epoch_loss_res

def print_train_eval_sup(num_classes, score_list_train, mask_list_train, print_num):

    if num_classes == 2:
        eval_list = evaluate(score_list_train, mask_list_train)
        print('| Train Thr: {:.4f}'.format(eval_list[0]).ljust(print_num, ' '), '|')
        print('| Train  Jc: {:.4f}'.format(eval_list[1]).ljust(print_num, ' '), '|')
        print('| Train  Dc: {:.4f}'.format(eval_list[2]).ljust(print_num, ' '), '|')
        train_m_jc = eval_list[1]

    else:
        eval_list = evaluate_multi(score_list_train, mask_list_train)

        np.set_printoptions(precision=4, suppress=True)
        print('| Train  Jc: {}'.format(eval_list[0]).ljust(print_num, ' '), '|')
        print('| Train  Dc: {}'.format(eval_list[2]).ljust(print_num, ' '), '|')
        print('| Train mJc: {:.4f}'.format(eval_list[1]).ljust(print_num, ' '), '|')
        print('| Train mDc: {:.4f}'.format(eval_list[3]).ljust(print_num, ' '), '|')
        train_m_jc = eval_list[1]

    return eval_list, train_m_jc

def print_train_eval_HFF(num_classes, score_list_train1, score_list_train2, mask_list_train, print_num):

    if num_classes == 2:
        eval_list1 = evaluate(score_list_train1, mask_list_train)
        eval_list2 = evaluate(score_list_train2, mask_list_train)
        print('| Train Thr 1: {:.4f}'.format(eval_list1[0]).ljust(print_num, ' '), '| Train Thr 2: {:.4f}'.format(eval_list2[0]).ljust(print_num, ' '), '|')
        print('| Train  Jc 1: {:.4f}'.format(eval_list1[1]).ljust(print_num, ' '), '| Train  Jc 2: {:.4f}'.format(eval_list2[1]).ljust(print_num, ' '), '|')
        print('| Train  Dc 1: {:.4f}'.format(eval_list1[2]).ljust(print_num, ' '), '| Train  Dc 2: {:.4f}'.format(eval_list2[2]).ljust(print_num, ' '), '|')
        print('| Train  Hd 1: {:.4f}'.format(eval_list1[3]).ljust(print_num, ' '), '| Train  Hd 2: {:.4f}'.format(eval_list2[3]).ljust(print_num, ' '), '|')

        train_m_dc1 = eval_list1[2]
        train_m_dc2 = eval_list2[2]

    else:
        eval_list1 = evaluate_multi(score_list_train1, mask_list_train)
        eval_list2 = evaluate_multi(score_list_train2, mask_list_train)
        np.set_printoptions(precision=4, suppress=True)

        print('| Train ET Dice 1: {:.4f}'.format(eval_list1[0]).ljust(print_num, ' '),
            '| Train ET Dice 2: {:.4f}'.format(eval_list2[0]).ljust(print_num, ' '), '|')
        print('| Train ET HD95 1: {:.4f}'.format(eval_list1[1]).ljust(print_num, ' '),
            '| Train ET HD95 2: {:.4f}'.format(eval_list2[1]).ljust(print_num, ' '), '|')
        print('| Train TC Dice 1: {:.4f}'.format(eval_list1[2]).ljust(print_num, ' '),
            '| Train TC Dice 2: {:.4f}'.format(eval_list2[2]).ljust(print_num, ' '), '|')
        print('| Train TC HD95 1: {:.4f}'.format(eval_list1[3]).ljust(print_num, ' '),
            '| Train TC HD95 2: {:.4f}'.format(eval_list2[3]).ljust(print_num, ' '), '|')
        print('| Train WT Dice 1: {:.4f}'.format(eval_list1[4]).ljust(print_num, ' '),
            '| Train WT Dice 2: {:.4f}'.format(eval_list2[4]).ljust(print_num, ' '), '|')
        print('| Train WT HD95 1: {:.4f}'.format(eval_list1[5]).ljust(print_num, ' '),
            '| Train WT HD95 2: {:.4f}'.format(eval_list2[5]).ljust(print_num, ' '), '|')
        train_m_dc1 = eval_list1[2]
        train_m_dc2 = eval_list2[2]

    return eval_list1, eval_list2, train_m_dc1, train_m_dc2

def print_val_eval_sup(num_classes, score_list_val, mask_list_val, print_num):
    if num_classes == 2:
        eval_list = evaluate(score_list_val, mask_list_val)
        print('| Val Thr: {:.4f}'.format(eval_list[0]).ljust(print_num, ' '), '|')
        print('| Val  Jc: {:.4f}'.format(eval_list[1]).ljust(print_num, ' '), '|')
        print('| Val  Dc: {:.4f}'.format(eval_list[2]).ljust(print_num, ' '), '|')
        val_m_jc = eval_list[1]
    else:
        eval_list = evaluate_multi(score_list_val, mask_list_val)
        np.set_printoptions(precision=4, suppress=True)
        print('| Val  Jc: {}  '.format(eval_list[0]).ljust(print_num, ' '), '|')
        print('| Val  Dc: {}  '.format(eval_list[2]).ljust(print_num, ' '), '|')
        print('| Val mJc: {:.4f}'.format(eval_list[1]).ljust(print_num, ' '), '|')
        print('| Val mDc: {:.4f}'.format(eval_list[3]).ljust(print_num, ' '), '|')
        val_m_jc = eval_list[1]
    return eval_list, val_m_jc

def print_val_eval(num_classes, score_list_val1, score_list_val2, mask_list_val, print_num):
    # if num_classes == 2:
    #     eval_list1 = evaluate(score_list_val1, mask_list_val)
    #     eval_list2 = evaluate(score_list_val2, mask_list_val)
    #     print('| Val Thr 1: {:.4f}'.format(eval_list1[0]).ljust(print_num, ' '), '| Val Thr 2: {:.4f}'.format(eval_list2[0]).ljust(print_num, ' '), '|')
    #     print('| Val  Jc 1: {:.4f}'.format(eval_list1[1]).ljust(print_num, ' '), '| Val  Jc 2: {:.4f}'.format(eval_list2[1]).ljust(print_num, ' '), '|')
    #     print('| Val  Dc 1: {:.4f}'.format(eval_list1[2]).ljust(print_num, ' '), '| Val  Dc 2: {:.4f}'.format(eval_list2[2]).ljust(print_num, ' '), '|')
    #     print('| Val  Hd 1: {:.4f}'.format(eval_list1[3]).ljust(print_num, ' '), '| Val  Hd 2: {:.4f}'.format(eval_list2[3]).ljust(print_num, ' '), '|')

    #     val_m_dc1 = eval_list1[2]
    #     val_m_dc2 = eval_list2[2]
    if num_classes == 2:
        # 如果样本数超过10，则按组评估；否则直接评估
        if score_list_val1.shape[0] > 10:
            eval_list1 = evaluate_groupwise_binary(score_list_val1, mask_list_val, group_size=10)
            eval_list2 = evaluate_groupwise_binary(score_list_val2, mask_list_val, group_size=10)
        else:
            eval_list1 = evaluate_multi_binary(score_list_val1, mask_list_val)
            eval_list2 = evaluate_multi_binary(score_list_val2, mask_list_val)
        
        np.set_printoptions(precision=4, suppress=True)
        print('| Val Dice LF: {:.4f}'.format(eval_list1[0]).ljust(print_num, ' '),
              '| Val Dice HF: {:.4f}'.format(eval_list2[0]).ljust(print_num, ' '), '|')
        print('| Val HD95 LF: {:.4f}'.format(eval_list1[1]).ljust(print_num, ' '),
              '| Val HD95 HF: {:.4f}'.format(eval_list2[1]).ljust(print_num, ' '), '|')
        
        val_m_dc1 = eval_list1[0]  # 可根据需求调整，这里取 Dice 值作为指标
        val_m_dc2 = eval_list2[0]
    else:
        if score_list_val1.shape[0] > 10:
            eval_list1 = evaluate_groupwise(score_list_val1, mask_list_val, group_size=10)
            eval_list2 = evaluate_groupwise(score_list_val2, mask_list_val, group_size=10)
        else:
            eval_list1 = evaluate_multi(score_list_val1, mask_list_val)
            eval_list2 = evaluate_multi(score_list_val2, mask_list_val)

        np.set_printoptions(precision=4, suppress=True)

        print('| Val ET Dice LF: {:.4f}'.format(eval_list1[0]).ljust(print_num, ' '),
            '| Val ET Dice HF: {:.4f}'.format(eval_list2[0]).ljust(print_num, ' '), '|')
        print('| Val ET HD95 LF: {:.4f}'.format(eval_list1[1]).ljust(print_num, ' '),
            '| Val ET HD95 HF: {:.4f}'.format(eval_list2[1]).ljust(print_num, ' '), '|')
        print('| Val TC Dice LF: {:.4f}'.format(eval_list1[2]).ljust(print_num, ' '),
            '| Val TC Dice HF: {:.4f}'.format(eval_list2[2]).ljust(print_num, ' '), '|')
        print('| Val TC HD95 LF: {:.4f}'.format(eval_list1[3]).ljust(print_num, ' '),
            '| Val TC HD95 HF: {:.4f}'.format(eval_list2[3]).ljust(print_num, ' '), '|')
        print('| Val WT Dice LF: {:.4f}'.format(eval_list1[4]).ljust(print_num, ' '),
            '| Val WT Dice HF: {:.4f}'.format(eval_list2[4]).ljust(print_num, ' '), '|')
        print('| Val WT HD95 LF: {:.4f}'.format(eval_list1[5]).ljust(print_num, ' '),
            '| Val WT HD95 HF: {:.4f}'.format(eval_list2[5]).ljust(print_num, ' '), '|')

        val_m_dc1 = eval_list1[2]
        val_m_dc2 = eval_list2[2]
    return eval_list1, eval_list2, val_m_dc1, val_m_dc2

def save_val_best_sup_2d(num_classes, best_list, model, score_list_val, name_list_val, eval_list, path_trained_model, path_seg_results, palette, model_name):

    if num_classes == 2:
        if best_list[1] < eval_list[1]:
            best_list = eval_list

            torch.save(model.state_dict(), os.path.join(path_trained_model, 'best_{}_Jc_{:.4f}.pth'.format(model_name, best_list[1])))

            score_list_val = torch.softmax(score_list_val, dim=1)
            pred_results = score_list_val[:, 1, :, :].cpu().numpy()
            pred_results[pred_results > eval_list[0]] = 1
            pred_results[pred_results <= eval_list[0]] = 0

            assert len(name_list_val) == pred_results.shape[0]
            for i in range(len(name_list_val)):
                color_results = Image.fromarray(pred_results[i].astype(np.uint8), mode='P')
                color_results.putpalette(palette)
                color_results.save(os.path.join(path_seg_results, name_list_val[i]))

    else:
        if best_list[1] < eval_list[1]:
            best_list = eval_list

            torch.save(model.state_dict(), os.path.join(path_trained_model, 'best_{}_Jc_{:.4f}.pth'.format(model_name, best_list[1])))

            pred_results = torch.max(score_list_val, 1)[1]
            pred_results = pred_results.cpu().numpy()

            assert len(name_list_val) == pred_results.shape[0]
            for i in range(len(name_list_val)):
                color_results = Image.fromarray(pred_results[i].astype(np.uint8), mode='P')
                color_results.putpalette(palette)
                color_results.save(os.path.join(path_seg_results, name_list_val[i]))

    return best_list

def save_val_best_sup_3d(num_classes, best_list, model, score_list_val, mask_list_val, eval_list, path_trained_model, path_seg_results, path_mask_results, model_name, format):

    if num_classes == 2:
        if best_list[1] < eval_list[1]:
            best_list = eval_list

            torch.save(model.state_dict(), os.path.join(path_trained_model, 'best_{}_Jc_{:.4f}.pth'.format(model_name, best_list[1])))

    else:
        if best_list[1] < eval_list[1]:
            best_list = eval_list

            torch.save(model.state_dict(), os.path.join(path_trained_model, 'best_{}_Jc_{:.4f}.pth'.format(model_name, best_list[1])))

    return best_list

def save_val_best_2d(num_classes, best_model, best_list, best_result, model1, model2, score_list_val_1, score_list_val_2, name_list_val, eval_list_1, eval_list_2, path_trained_model, path_seg_results, palette):

    if eval_list_1[1] < eval_list_2[1]:
        if best_list[1] < eval_list_2[1]:

            best_model = model2
            best_list = eval_list_2
            best_result = 'Result2'
            if not os.path.exists(path_trained_model):
                os.mkdir(path_trained_model)
            torch.save(model2.state_dict(), os.path.join(path_trained_model, 'best_{}_Jc_{:.4f}.pth'.format('result2', best_list[1])))

            if num_classes == 2:
                score_list_val_2 = torch.softmax(score_list_val_2, dim=1)
                pred_results = score_list_val_2[:, 1, ...].cpu().numpy()
                pred_results[pred_results > eval_list_2[0]] = 1
                pred_results[pred_results <= eval_list_2[0]] = 0
            else:
                pred_results = torch.max(score_list_val_2, 1)[1]
                pred_results = pred_results.cpu().numpy()

            assert len(name_list_val) == pred_results.shape[0]
            for i in range(len(name_list_val)):
            # color_results = Image.fromarray(pred_results[i].astype(np.uint8), mode='P')
                color_results = Image.fromarray(pred_results[i].astype(np.uint8), mode='P')
                color_results.putpalette(palette)
            # color_results.save(os.path.join(path_seg_results, name_list_val[i]))
                color_results.save(os.path.join(path_seg_results, name_list_val[i]))
        else:
            best_model = best_model
            best_list = best_list
            best_result = best_result

    else:
        if best_list[1] < eval_list_1[1]:

            best_model = model1
            best_list = eval_list_1
            best_result = 'Result1'

            torch.save(model1.state_dict(), os.path.join(path_trained_model, 'best_{}_Jc_{:.4f}.pth'.format('result1', best_list[1])))

            if num_classes == 2:
                score_list_val_1 = torch.softmax(score_list_val_1, dim=1)
                pred_results = score_list_val_1[:, 1, ...].cpu().numpy()
                pred_results[pred_results > eval_list_1[0]] = 1
                pred_results[pred_results <= eval_list_1[0]] = 0
            else:
                pred_results = torch.max(score_list_val_1, 1)[1]
                pred_results = pred_results.cpu().numpy()
            # print(name_list_val,len(name_list_val))
            # print(pred_results.shape)
            assert len(name_list_val) == pred_results.shape[0]
            for i in range(len(name_list_val)):
                color_results = Image.fromarray(pred_results[i].astype(np.uint8), mode='P')
            # color_results = Image.fromarray(pred_results[0].astype(np.uint8), mode='P')
                color_results.putpalette(palette)
            # color_results.save(os.path.join(path_seg_results, name_list_val[i]))
                color_results.save(os.path.join(path_seg_results, name_list_val[i]))
        else:
            best_model = best_model
            best_list = best_list
            best_result = best_result


    return best_list, best_model, best_result


def save_val_best_3d(num_classes, best_model, best_list, best_result, model1, model2, score_list_val_1, score_list_val_2, mask_list_val, eval_list_1, eval_list_2, path_trained_model, path_seg_results, path_mask_results, format):

    if eval_list_1[2] < eval_list_2[2]:
        if best_list[2] < eval_list_2[2]:

            best_model = model2
            best_list = eval_list_2
            best_result = 'Result2'

            torch.save(model2.state_dict(), os.path.join(path_trained_model, 'best_{}_Dc_{:.4f}.pth'.format('result2', best_list[2])))

        else:
            best_model = best_model
            best_list = best_list
            best_result = best_result

    else:
        if best_list[2] < eval_list_1[2]:

            best_model = model1
            best_list = eval_list_1
            best_result = 'Result1'

            torch.save(model1.state_dict(), os.path.join(path_trained_model, 'best_{}_Dc_{:.4f}.pth'.format('result1', best_list[2])))

        else:
            best_model = best_model
            best_list = best_list
            best_result = best_result

    return best_list, best_model, best_result

# def save_val_best_3d_m(num_classes, best_model, best_list, best_result,
#                        model1, model2, score_list_val_1, score_list_val_2,
#                        mask_list_val, eval_list_1, eval_list_2,
#                        path_trained_model, path_seg_results, path_mask_results):
#     """
#     Parameter description: 
#     - For multi-classification (for example, 4 classes are mapped to 0,1,2,3), the format of eval_list_* is: 
#     [dice_et, hd95_et, dice_tc, hd95_tc, dice_wt, hd95_wt]
#     Here, the Dice average of the three structures (ET, TC, WT) is used as the evaluation indicator. As long as there is an improvement, the model is saved. 
#     The Dice value of each structure is written in the file name. 
#     - For binary classification, the mask has been mapped to {0,1}, and the format of eval_list_* is (dice, hd95), and dice is directly used as the indicator.
#     """
#     if num_classes == 2:
#         # 二分类情况下，只取 dice 指标
#         avg_dice1 = eval_list_1[0]
#         avg_dice2 = eval_list_2[0]
#         best_metric = best_list[0] if best_list is not None else -1

#         if avg_dice1 >= avg_dice2:
#             if avg_dice1 > best_metric:
#                 best_model = model1
#                 best_list = eval_list_1
#                 best_result = 'Result1'
#                 filename = 'best_Result1_dice_{:.4f}.pth'.format(eval_list_1[0])
#                 torch.save(model1.state_dict(), os.path.join(path_trained_model, filename))
#         else:
#             if avg_dice2 > best_metric:
#                 best_model = model2
#                 best_list = eval_list_2
#                 best_result = 'Result2'
#                 filename = 'best_Result2_dice_{:.4f}.pth'.format(eval_list_2[0])
#                 torch.save(model2.state_dict(), os.path.join(path_trained_model, filename))
#     else:
#         # 多分类情况：采用 ET、TC、WT 三个结构的 Dice 平均值
#         avg_dice1 = (eval_list_1[0] + eval_list_1[2] + eval_list_1[4]) / 3.0
#         avg_dice2 = (eval_list_2[0] + eval_list_2[2] + eval_list_2[4]) / 3.0
#         best_metric = (best_list[0] + best_list[2] + best_list[4]) / 3.0 if best_list is not None else -1

#         if avg_dice1 >= avg_dice2:
#             if avg_dice1 > best_metric:
#                 best_model = model1
#                 best_list = eval_list_1
#                 best_result = 'Result1'
#                 filename = 'best_Result1_et_{:.4f}_tc_{:.4f}_wt_{:.4f}.pth'.format(
#                             eval_list_1[0], eval_list_1[2], eval_list_1[4])
#                 torch.save(model1.state_dict(), os.path.join(path_trained_model, filename))
#         else:
#             if avg_dice2 > best_metric:
#                 best_model = model2
#                 best_list = eval_list_2
#                 best_result = 'Result2'
#                 filename = 'best_Result2_et_{:.4f}_tc_{:.4f}_wt_{:.4f}.pth'.format(
#                             eval_list_2[0], eval_list_2[2], eval_list_2[4])
#                 torch.save(model2.state_dict(), os.path.join(path_trained_model, filename))
                
#     return best_list, best_model, best_result


def save_val_best_3d_m(num_classes, best_model, best_list, best_result,
                       model1, model2, score_list_val_1, score_list_val_2,
                       mask_list_val, eval_list_1, eval_list_2,
                       path_trained_model,):
    """
    - 二分类时：eval_list_* = [dice, hd95]
    - 多分类(all)时：eval_list_* = [dice_et, hd95_et, dice_tc, hd95_tc, dice_wt, hd95_wt]
    我们用 Dice 平均来决定是否保存新最佳模型。
    """
    # —— 二分类逻辑 —— #
    if num_classes == 2:
        avg_dice1   = eval_list_1[0]
        avg_dice2   = eval_list_2[0]
        best_metric = best_list[0] if (best_list and len(best_list) >= 1) else -1.0

        if avg_dice1 >= avg_dice2:
            if avg_dice1 > best_metric:
                best_model  = model1
                best_list   = eval_list_1
                best_result = 'Result1'
                filename    = f'best_Result1_dice_{eval_list_1[0]:.4f}.pth'
                torch.save(model1.state_dict(), os.path.join(path_trained_model, filename))
        else:
            if avg_dice2 > best_metric:
                best_model  = model2
                best_list   = eval_list_2
                best_result = 'Result2'
                filename    = f'best_Result2_dice_{eval_list_2[0]:.4f}.pth'
                torch.save(model2.state_dict(), os.path.join(path_trained_model, filename))

    # —— 多分类(all)逻辑 —— #
    else:
        # print_val_eval 在 all 模式下返回长度 6 的列表：
        #   [dice_et, hd95_et, dice_tc, hd95_tc, dice_wt, hd95_wt]
        metric_idxs = [0, 2, 4]  # 对应 et, tc, wt 的 Dice 索引
        # 计算平均 Dice
        avg_dice1 = sum(eval_list_1[i] for i in metric_idxs) / len(metric_idxs)
        avg_dice2 = sum(eval_list_2[i] for i in metric_idxs) / len(metric_idxs)

        # 安全获取历史最佳：必须至少有 6 个元素，否则认为无历史最佳
        if not best_list or len(best_list) < 6:
            best_metric = -1.0
        else:
            best_metric = sum(best_list[i] for i in metric_idxs) / len(metric_idxs)

        # 比较并保存
        if avg_dice1 >= avg_dice2:
            if avg_dice1 > best_metric:
                best_model  = model1
                best_list   = eval_list_1
                best_result = 'Result1'
                et, tc, wt = (eval_list_1[i] for i in metric_idxs)
                filename = f'best_Result1_et_{et:.4f}_tc_{tc:.4f}_wt_{wt:.4f}.pth'
                torch.save(model1.state_dict(), os.path.join(path_trained_model, filename))
        else:
            if avg_dice2 > best_metric:
                best_model  = model2
                best_list   = eval_list_2
                best_result = 'Result2'
                et, tc, wt = (eval_list_2[i] for i in metric_idxs)
                filename = f'best_Result2_et_{et:.4f}_tc_{tc:.4f}_wt_{wt:.4f}.pth'
                torch.save(model2.state_dict(), os.path.join(path_trained_model, filename))

    return best_list, best_model, best_result

def draw_pred_sup(num_classes, mask_train_sup, mask_val, pred_train_sup, outputs_val, train_eval_list, val_eval_list):


    mask_image_train_sup = mask_train_sup[0, :, :].data.cpu().numpy()
    mask_image_val = mask_val[0, :, :].data.cpu().numpy()

    if num_classes == 2:
        pred_image_train_sup = pred_train_sup[0, 1, :, :].data.cpu().numpy()
        pred_image_train_sup[pred_image_train_sup > train_eval_list[0]] = 1
        pred_image_train_sup[pred_image_train_sup <= train_eval_list[0]] = 0

        pred_image_val = outputs_val[0, 1, :, :].data.cpu().numpy()
        pred_image_val[pred_image_val > val_eval_list[0]] = 1
        pred_image_val[pred_image_val <= val_eval_list[0]] = 0

    else:
        pred_image_train_sup = torch.max(pred_train_sup, 1)[1]
        pred_image_train_sup = pred_image_train_sup[0, :, :].cpu().numpy()

        pred_image_val = torch.max(outputs_val, 1)[1]
        pred_image_val = pred_image_val[0, :, :].cpu().numpy()

    return mask_image_train_sup, pred_image_train_sup, mask_image_val, pred_image_val


def draw_pred_HFF(num_classes, mask_train, mask_val, pred_train_sup1, pred_train_sup2, outputs_val1, outputs_val2, train_eval_list1, train_eval_list2, val_eval_list1, val_eval_list2):


    mask_image_train_sup = mask_train[0, :, :].data.cpu().numpy()
    mask_image_val = mask_val[0, :, :].data.cpu().numpy()

    if num_classes == 2:

        pred_image_train_sup1 = pred_train_sup1[0, 1, :, :].data.cpu().numpy()
        pred_image_train_sup1[pred_image_train_sup1 > train_eval_list1[0]] = 1
        pred_image_train_sup1[pred_image_train_sup1 <= train_eval_list1[0]] = 0

        pred_image_train_sup2 = pred_train_sup2[0, 1, :, :].data.cpu().numpy()
        pred_image_train_sup2[pred_image_train_sup2 > train_eval_list2[0]] = 1
        pred_image_train_sup2[pred_image_train_sup2 <= train_eval_list2[0]] = 0

        pred_image_val1 = outputs_val1[0, 1, :, :].data.cpu().numpy()
        pred_image_val1[pred_image_val1 > val_eval_list1[0]] = 1
        pred_image_val1[pred_image_val1 <= val_eval_list1[0]] = 0

        pred_image_val2 = outputs_val2[0, 1, :, :].data.cpu().numpy()
        pred_image_val2[pred_image_val2 > val_eval_list2[0]] = 1
        pred_image_val2[pred_image_val2 <= val_eval_list2[0]] = 0
    else:

        pred_image_train_sup1 = torch.max(pred_train_sup1, 1)[1]
        pred_image_train_sup1 = pred_image_train_sup1[0, :, :].cpu().numpy()

        pred_image_train_sup2 = torch.max(pred_train_sup2, 1)[1]
        pred_image_train_sup2 = pred_image_train_sup2[0, :, :].cpu().numpy()

        pred_image_val1 = torch.max(outputs_val1, 1)[1]
        pred_image_val1 = pred_image_val1[0, :, :].cpu().numpy()

        pred_image_val2 = torch.max(outputs_val2, 1)[1]
        pred_image_val2 = pred_image_val2[0, :, :].cpu().numpy()

    return mask_image_train_sup, pred_image_train_sup1, pred_image_train_sup2, mask_image_val, pred_image_val1, pred_image_val2

def draw_pred_MT(num_classes, mask_train, mask_val, pred_train_sup1, outputs_val1, outputs_val2, train_eval_list1, val_eval_list1, val_eval_list2):


    mask_image_train_sup = mask_train[0, :, :].data.cpu().numpy()
    mask_image_val = mask_val[0, :, :].data.cpu().numpy()

    if num_classes == 2:

        pred_image_train_sup1 = pred_train_sup1[0, 1, :, :].data.cpu().numpy()
        pred_image_train_sup1[pred_image_train_sup1 > train_eval_list1[0]] = 1
        pred_image_train_sup1[pred_image_train_sup1 <= train_eval_list1[0]] = 0

        pred_image_val1 = outputs_val1[0, 1, :, :].data.cpu().numpy()
        pred_image_val1[pred_image_val1 > val_eval_list1[0]] = 1
        pred_image_val1[pred_image_val1 <= val_eval_list1[0]] = 0

        pred_image_val2 = outputs_val2[0, 1, :, :].data.cpu().numpy()
        pred_image_val2[pred_image_val2 > val_eval_list2[0]] = 1
        pred_image_val2[pred_image_val2 <= val_eval_list2[0]] = 0
    else:

        pred_image_train_sup1 = torch.max(pred_train_sup1, 1)[1]
        pred_image_train_sup1 = pred_image_train_sup1[0, :, :].cpu().numpy()

        pred_image_val1 = torch.max(outputs_val1, 1)[1]
        pred_image_val1 = pred_image_val1[0, :, :].cpu().numpy()

        pred_image_val2 = torch.max(outputs_val2, 1)[1]
        pred_image_val2 = pred_image_val2[0, :, :].cpu().numpy()

    return mask_image_train_sup, pred_image_train_sup1, mask_image_val, pred_image_val1, pred_image_val2


def print_best_sup(num_classes, best_val_list, print_num):
    if num_classes == 2:
        print('| Best Val Thr: {:.4f}'.format(best_val_list[0]).ljust(print_num, ' '), '|')
        print('| Best Val  Jc: {:.4f}'.format(best_val_list[1]).ljust(print_num, ' '), '|')
        print('| Best Val  Dc: {:.4f}'.format(best_val_list[2]).ljust(print_num, ' '), '|')
    else:
        np.set_printoptions(precision=4, suppress=True)
        print('| Best Val  Jc: {}'.format(best_val_list[0]).ljust(print_num, ' '), '|')
        print('| Best Val  Dc: {}'.format(best_val_list[2]).ljust(print_num, ' '), '|')
        print('| Best Val mJc: {:.4f}'.format(best_val_list[1]).ljust(print_num, ' '), '|')
        print('| Best Val mDc: {:.4f}'.format(best_val_list[3]).ljust(print_num, ' '), '|')

def print_best(num_classes, best_val_list, best_model, best_result, path_trained_model, print_num):
    if num_classes == 2:

        torch.save(best_model.state_dict(), os.path.join(path_trained_model, 'best_Jc_{:.4f}.pth'.format(best_val_list[1])))

        print('| Best  Result: {}'.format(best_result).ljust(print_num, ' '), '|')
        print('| Best Val Thr: {:.4f}'.format(best_val_list[0]).ljust(print_num, ' '), '|')
        print('| Best Val  Jc: {:.4f}'.format(best_val_list[1]).ljust(print_num, ' '), '|')
        print('| Best Val  Dc: {:.4f}'.format(best_val_list[2]).ljust(print_num, ' '), '|')
        print('| Best Val  Hd: {:.4f}'.format(best_val_list[3]).ljust(print_num, ' '), '|')

    else:

        torch.save(best_model.state_dict(), os.path.join(path_trained_model, 'best_Jc_{:.4f}.pth'.format(best_val_list[1])))

        np.set_printoptions(precision=4, suppress=True)
        print('| Best  Result: {}'.format(best_result).ljust(print_num, ' '), '|')
        print('| Best Val  Jc: {}'.format(best_val_list[0]).ljust(print_num, ' '), '|')
        print('| Best Val  Dc: {}'.format(best_val_list[2]).ljust(print_num, ' '), '|')
        print('| Best Val mJc: {:.4f}'.format(best_val_list[1]).ljust(print_num, ' '), '|')
        print('| Best Val mDc: {:.4f}'.format(best_val_list[3]).ljust(print_num, ' '), '|')

def print_test_eval(num_classes, score_list_test, mask_list_test, print_num):
    if num_classes == 2:
        eval_list = evaluate(score_list_test, mask_list_test)
        print('| Test Thr: {:.4f}'.format(eval_list[0]).ljust(print_num, ' '), '|')
        print('| Test  Jc: {:.4f}'.format(eval_list[1]).ljust(print_num, ' '), '|')
        print('| Test  Dc: {:.4f}'.format(eval_list[2]).ljust(print_num, ' '), '|')
    else:
        eval_list = evaluate_multi(score_list_test, mask_list_test)
        np.set_printoptions(precision=4, suppress=True)
        print('| Test  Jc: {}  '.format(eval_list[0]).ljust(print_num, ' '), '|')
        print('| Test  Dc: {}  '.format(eval_list[2]).ljust(print_num, ' '), '|')
        print('| Test mJc: {:.4f}'.format(eval_list[1]).ljust(print_num, ' '), '|')
        print('| Test mDc: {:.4f}'.format(eval_list[3]).ljust(print_num, ' '), '|')

    return eval_list


def save_test_2d(num_classes, score_list_test, name_list_test, threshold, path_seg_results, palette):

    if num_classes == 2:
        score_list_test = torch.softmax(score_list_test, dim=1)
        pred_results = score_list_test[:, 1, ...].cpu().numpy()
        pred_results[pred_results > threshold] = 1
        pred_results[pred_results <= threshold] = 0

        assert len(name_list_test) == pred_results.shape[0]

        for i in range(len(name_list_test)):
            color_results = Image.fromarray(pred_results[i].astype(np.uint8), mode='P')
        #     color_results = Image.fromarray(pred_results[0].astype(np.uint8), mode='P')
            color_results.putpalette(palette)
        # color_results.save(os.path.join(path_seg_results, name_list_test[i]))
            color_results.save(os.path.join(path_seg_results, name_list_test[i]))

    else:
        pred_results = torch.max(score_list_test, 1)[1]
        pred_results = pred_results.cpu().numpy()

        assert len(name_list_test) == pred_results.shape[0]

        for i in range(len(name_list_test)):
            color_results = Image.fromarray(pred_results[i].astype(np.uint8), mode='P')
            color_results.putpalette(palette)
            color_results.save(os.path.join(path_seg_results, name_list_test[i]))

def save_test_3d(num_classes, score_test, name_test, threshold, path_seg_results, affine):

    if num_classes == 2:
        score_list_test = torch.softmax(score_test, dim=0)
        pred_results = score_list_test[1, ...].cpu()
        pred_results[pred_results > threshold] = 1
        pred_results[pred_results <= threshold] = 0

        pred_results = pred_results.type(torch.uint8)

        output_image = tio.ScalarImage(tensor=pred_results.unsqueeze(0), affine=affine)
        output_image.save(os.path.join(path_seg_results, name_test))

    else:
        pred_results = torch.max(score_test, 0)[1]
        pred_results = pred_results.cpu()
        pred_results = pred_results.type(torch.uint8)

        output_image = tio.ScalarImage(tensor=pred_results.unsqueeze(0), affine=affine)
        output_image.save(os.path.join(path_seg_results, name_test))




import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import sys
from torch.nn.modules.loss import _Loss
import torch.fft



class MixSoftmaxCrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, aux=True, aux_weight=0.2, ignore_index=-1, **kwargs):
        super(MixSoftmaxCrossEntropyLoss, self).__init__(ignore_index=ignore_index)
        self.aux = aux
        self.aux_weight = aux_weight

    def _aux_forward(self, output, target, **kwargs):
        # *preds, target = tuple(inputs)

        loss = super(MixSoftmaxCrossEntropyLoss, self).forward(output[0], target)
        for i in range(1, len(output)):
            aux_loss = super(MixSoftmaxCrossEntropyLoss, self).forward(output[i], target)
            loss += self.aux_weight * aux_loss
        return loss


class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """

    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target, valid_mask):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1).float()
        valid_mask = valid_mask.contiguous().view(valid_mask.shape[0], -1).float()

        num = torch.sum(torch.mul(predict, target) * valid_mask, dim=1) * 2 + self.smooth
        den = torch.sum((predict.pow(self.p) + target.pow(self.p)) * valid_mask, dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))



class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input"""

    def __init__(self, weight=None, aux=False, aux_weight=0.4, ignore_index=-1, cn=2,**kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index
        self.aux = aux
        self.cn= cn
        self.aux_weight = aux_weight

    def _base_forward(self, predict, target, valid_mask):

        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        predict = F.softmax(predict, dim=1)

        for i in range(target.shape[-1]):
            if i != self.ignore_index:
                dice_loss = dice(predict[:, i], target[..., i], valid_mask)
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    dice_loss *= self.weights[i]
                total_loss += dice_loss

        return total_loss / target.shape[-1]

    def _aux_forward(self, output, target, **kwargs):
        # valid_mask = (target != self.ignore_index).long()
        #     # 这里将 target 转换为 one-hot，明确指定类别数为 4
        # target_one_hot = F.one_hot(torch.clamp_min(target, 0), num_classes=4)
        # target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()
        # return self._base_forward(output, target_one_hot, valid_mask)
        # *preds, target = tuple(inputs)
        valid_mask = (target != self.ignore_index).long()
        # bras 4分类修改
        target_one_hot = F.one_hot(torch.clamp_min(target, 0),num_classes=4)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()

        loss = self._base_forward(output[0], target_one_hot, valid_mask)
        for i in range(1, len(output)):
            aux_loss = self._base_forward(output[i], target_one_hot, valid_mask)
            loss += self.aux_weight * aux_loss
        return loss
    

    def forward(self, output, target):
        # preds, target = tuple(inputs)
        # inputs = tuple(list(preds) + [target])
        if self.aux:
            return self._aux_forward(output, target)
        else:
            # mask = target != self.ignore_index
            # valid_mask = mask.any(dim=3).long()

            # 3/6修改
            # valid_mask = (target != self.ignore_index).long()
            # target_one_hot = F.one_hot(torch.clamp_min(target, 0))
            # return self._base_forward(output, target_one_hot, valid_mask)
        
            valid_mask = (target != self.ignore_index).long()
            # 这里将 target 转换为 one-hot，明确指定类别数为 4
            target_one_hot = F.one_hot(torch.clamp_min(target, 0), num_classes=self.cn)
            # target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()
            return self._base_forward(output, target_one_hot, valid_mask)

class BCELossBoud(nn.Module):
    def __init__(self, num_classes, weight=None, ignore_index=None, **kwargs):
        super(BCELossBoud, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.criterion = nn.BCEWithLogitsLoss()

    def weighted_BCE_cross_entropy(self, output, target, weights = None):
        if weights is not None:
            assert len(weights) == 2
            output = torch.clamp(output, min=1e-3, max=1-1e-3)
            bce = weights[1] * (target * torch.log(output)) + weights[0] * ((1-target) * torch.log((1-output)))
        else:
            output = torch.clamp(output, min=1e-3, max=1 - 1e-3)
            bce = target * torch.log(output) + (1-target) * torch.log((1-output))
        return torch.neg(torch.mean(bce))

    def forward(self, predict, target):

        target_one_hot = F.one_hot(torch.clamp_min(target, 0), num_classes=self.num_classes).permute(0, 4, 1, 2, 3)
        predict = torch.softmax(predict, 1)

        bs, category, depth, width, heigt = target_one_hot.shape
        bce_loss = []
        for i in range(predict.shape[1]):
            pred_i = predict[:,i]
            targ_i = target_one_hot[:,i]
            tt = np.log(depth * width * heigt / (target_one_hot[:, i].cpu().data.numpy().sum()+1))
            bce_i = self.weighted_BCE_cross_entropy(pred_i, targ_i, weights=[1, tt])
            bce_loss.append(bce_i)

        bce_loss = torch.stack(bce_loss)
        total_loss = bce_loss.mean()
        return total_loss


def segmentation_loss(loss='CE', aux=False, cn=2, **kwargs):

    if loss == 'dice' or loss == 'DICE':
        seg_loss = DiceLoss(aux=aux,cn=cn, **kwargs)
    elif loss == 'crossentropy' or loss == 'CE':
        seg_loss = MixSoftmaxCrossEntropyLoss(aux=aux, **kwargs)
    elif loss == 'bce':
        seg_loss = nn.BCELoss(size_average=True)
    elif loss == 'bcebound':
        seg_loss = BCELossBoud(num_classes=kwargs['num_classes'])
    elif loss == 'ff':
        seg_loss= FocalFrequencyLoss()
    else:
        print('sorry, the loss you input is not supported yet')
        sys.exit()

    return seg_loss



class FocalFrequencyLoss(nn.Module):

    def __init__(self, loss_weight=1.0, alpha=1.0, patch_factor=1, ave_spectrum=False, log_matrix=False, batch_matrix=False):
        super(FocalFrequencyLoss, self).__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.patch_factor = patch_factor
        self.ave_spectrum = ave_spectrum
        self.log_matrix = log_matrix
        self.batch_matrix = batch_matrix

    def tensor2freq(self, x):
        # crop image patches

        # _, d, h, w = x.shape
        _,_, d, h, w = x.shape
        assert d % self.patch_factor == 0 and h % self.patch_factor == 0 and w % self.patch_factor == 0, (
            'Patch factor should be divisible by image depth, height and width')
        patch_list = []
        patch_size_d = d // self.patch_factor
        patch_size_h = h // self.patch_factor
        patch_size_w = w // self.patch_factor
        for i in range(self.patch_factor):
            for j in range(self.patch_factor):
                for k in range(self.patch_factor):
                    patch = x[:,
                            i * patch_size_d: (i + 1) * patch_size_d,
                            j * patch_size_h: (j + 1) * patch_size_h,
                            k * patch_size_w: (k + 1) * patch_size_w]
                    patch_list.append(patch)
        # stack to patch tensor
        y = torch.stack(patch_list, dim=0)
        freq = torch.fft.fftn(y, dim=(2, 3, 4), norm='ortho')
        freq = torch.stack([freq.real, freq.imag], -1)

        return freq

    def loss_formulation(self, recon_freq, real_freq, matrix=None):
        # spectrum weight matrix
        if matrix is not None:
            # if the matrix is predefined
            weight_matrix = matrix.detach()
        else:
            # if the matrix is calculated online: continuous, dynamic, based on current Euclidean distance
            matrix_tmp = (recon_freq - real_freq) ** 2
            matrix_tmp = torch.sqrt(matrix_tmp[..., 0] + matrix_tmp[..., 1]) ** self.alpha

            # whether to adjust the spectrum weight matrix by logarithm
            if self.log_matrix:
                matrix_tmp = torch.log(matrix_tmp + 1.0)

            # whether to calculate the spectrum weight matrix using batch-based statistics
            if self.batch_matrix:
                matrix_tmp = matrix_tmp / matrix_tmp.max()
            else:
                max_val = matrix_tmp.max(dim=-1, keepdim=True).values.max(dim=-2, keepdim=True).values.max(dim=-3,
                                                                                                           keepdim=True).values
                matrix_tmp = matrix_tmp / max_val

            matrix_tmp[torch.isnan(matrix_tmp)] = 0.0
            matrix_tmp = torch.clamp(matrix_tmp, min=0.0, max=1.0)
            weight_matrix = matrix_tmp.clone().detach()

        assert weight_matrix.min().item() >= 0 and weight_matrix.max().item() <= 1, (
            'The values of spectrum weight matrix should be in the range [0, 1], '
            'but got Min: %.10f Max: %.10f' % (weight_matrix.min().item(), weight_matrix.max().item()))

        # frequency distance using (squared) Euclidean distance
        tmp = (recon_freq - real_freq) ** 2
        freq_distance = tmp[..., 0] + tmp[..., 1]

        # dynamic spectrum weighting (Hadamard product)
        loss = weight_matrix * freq_distance
        return torch.mean(loss)

    # def forward(self, predict, target, matrix=None, **kwargs):
    #     total_loss = 0
    #     predict = F.softmax(predict, dim=1)
    #     target = F.one_hot(torch.clamp_min(target, 0))
    #
    #     # whether to use minibatch average spectrum
    #     for i in range(target.shape[-1]):
    #         pred_freq = self.tensor2freq(predict[:, i])
    #         target_freq = self.tensor2freq(target[..., i])
    #         if self.ave_spectrum:
    #             pred_freq = torch.mean(pred_freq, 0, keepdim=True)
    #             target_freq = torch.mean(target_freq, 0, keepdim=True)
    #         loss = self.loss_formulation(pred_freq, target_freq, matrix) * self.loss_weight
    #         total_loss += loss
    #
    #     # calculate focal frequency loss
    #     return total_loss / target.shape[-1]
    def forward(self, pred, target, matrix=None, **kwargs):
        """Forward function to calculate focal frequency loss.

        Args:
            pred (torch.Tensor): of shape (N, C, H, W). Predicted tensor.
            target (torch.Tensor): of shape (N, C, H, W). Target tensor.
            matrix (torch.Tensor, optional): Element-wise spectrum weight matrix.
                Default: None (If set to None: calculated online, dynamic).
        """
        pred_freq = self.tensor2freq(pred)
        target_freq = self.tensor2freq(target)

        # whether to use minibatch average spectrum
        if self.ave_spectrum:
            pred_freq = torch.mean(pred_freq, 0, keepdim=True)
            target_freq = torch.mean(target_freq, 0, keepdim=True)

        # calculate focal frequency loss
        return self.loss_formulation(pred_freq, target_freq, matrix) * self.loss_weight


if __name__=="__main__":
    a = torch.randn(1, 1, 128, 128, 128).cuda()
    b = torch.randn(1, 1, 128, 128, 128).cuda()

    m=FocalFrequencyLoss().cuda()
    output=m(a,b)
    print(output)

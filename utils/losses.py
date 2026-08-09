import torch
import torch.nn.functional as F
import torch.nn as nn



class L1WTVLossFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pred, target, seg, alpha):
        l1_loss = F.l1_loss(pred, target, reduction='mean')

        grad_sr_x = F.pad(torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:]), pad=(0, 1, 0, 0))
        grad_sr_y = F.pad(torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :]), pad=(0, 0, 0, 1))

        grad_seg_x = F.pad(torch.abs(seg[:, :, :, :-1] - seg[:, :, :, 1:]), pad=(0, 1, 0, 0))
        grad_seg_y = F.pad(torch.abs(seg[:, :, :-1, :] - seg[:, :, 1:, :]), pad=(0, 0, 0, 1))

        grad_sr = torch.stack(tensors=[grad_sr_x, grad_sr_y], dim=0)
        grad_seg = torch.stack(tensors=[grad_seg_x, grad_seg_y], dim=0)

        wtv_loss = torch.mean(grad_seg / (grad_sr + 1e-5))

        ctx.save_for_backward(pred, target, grad_sr, grad_seg)
        ctx.alpha = alpha

        return l1_loss + alpha * wtv_loss

    @staticmethod
    def backward(ctx, grad_output):
        pred, target, grad_sr, grad_seg = ctx.saved_tensors
        alpha = ctx.alpha

        grad_l1 = torch.sign(pred - target) / pred.numel()

        grad_wtv = -grad_seg / ((grad_sr + 1e-5) ** 2)
        grad_wtv /= pred.numel()

        l1_max = grad_l1.abs().max()
        grad_wtv = torch.clamp(grad_wtv, min=-l1_max.item(), max=l1_max.item())

        grad_pred = grad_output * (grad_l1 + alpha * grad_wtv)

        return grad_pred, None, None, None


class L1WTVLoss(nn.Module):
    def __init__(self, loss_weight=1.0, wtv_weight=0.1):
        super().__init__()
        self.loss_weight = loss_weight
        self.wtv_weight = wtv_weight

    def forward(self, pred, target, seg):
        loss = L1WTVLossFunction.apply(pred, target, seg, self.wtv_weight)
        return self.loss_weight * loss

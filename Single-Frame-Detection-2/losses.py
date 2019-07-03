import torch
from torch import nn

def _sigmoid(x):
    x = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
    return x

# 2）正确检测且精度满足预定要求[即有且仅有1个检测结果位于3×3的标注框内（含）]，每个坐标位置得1分；
# 3）正确检测但精度不满足预定要求[即有且仅有1个检测结果位于3×3的标注框外且位于9×9的标注框内（含）]，每个坐标位置得0分；
# 4）漏检[指9×9的标注框内（含）无检测结果]，每个坐标位置减1分；
# 5）虚警[指在9×9的标注框外出现检测结果，或1个标注框内出现多余1个的检测结果]，每个坐标位置减2分；

# 虽然评价指标要求为3*3，但在9*9对检测有一定帮助，起码不扣分，# 因此，这里我们参考CornerNet，
# 将9×9范围内的检测均视为postive，但对3*3以外9×9以内的检测施加一定的惩罚，离中心越远损失越大。
# 同时，考虑到9*9以外扣分较多（2分），我们适当增加negative的惩罚权重。

def weighted_focal_loss(pred_heat_map, gt, neg_weights=2):
    pred = _sigmoid(pred_heat_map)

    pos_inds = gt.eq(1)
    neg_inds = gt.lt(1)

    pos_weights = torch.pow(1 - gt[pos_inds], 4) # to update
    # neg_weights = torch.pow(1 - gt[neg_inds], 4)

    pos_pred = pred[pos_inds]
    neg_pred = pred[neg_inds]
    pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2)*pos_weights
    neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, 2) * neg_weights

    num_pos  = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if pos_pred.nelement() == 0:
        loss =  - neg_loss
    else:
        loss =  - (pos_loss + neg_loss) / num_pos

    return loss.unsqueeze(0)

    

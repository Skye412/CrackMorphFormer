import torch
import torch.nn as nn
import torch.nn.functional as F

class BoundaryRelaxedLoss(nn.Module):
    """
    抗标注噪声的边界松弛损失 (Boundary-Relaxed Loss)
    """
    def __init__(self, alpha=0.1, dilation_kernel=5, ignore_index=255):
        super(BoundaryRelaxedLoss, self).__init__()
        self.alpha = alpha                # 缓冲带内的假阳性豁免权重 (调低惩罚)
        self.kernel_size = dilation_kernel# 膨胀缓冲带的宽度
        self.ignore_index = ignore_index  # 255 标签隔离

    def forward(self, pred_logits, target):
        # target: [B, 1, H, W]
        # 1. 过滤 255 忽略标签 (剥落、泛碱等非裂缝病害不计算 Loss)
        valid_mask = (target != self.ignore_index).float()
        clean_target = target.clone().float()
        clean_target[target == self.ignore_index] = 0.0 
        
        # 2. 形态学膨胀，制造宽容缓冲带
        padding = self.kernel_size // 2
        dilated_target = F.max_pool2d(
            clean_target, kernel_size=self.kernel_size, stride=1, padding=padding
        )
        
        # 缓冲带 = 膨胀后的标签 - 原本的标签
        m_relax = dilated_target - clean_target
        
        # 3. 计算基础的二值交叉熵 (BCE)
        bce_loss = F.binary_cross_entropy_with_logits(pred_logits, clean_target, reduction='none')
        
        # 4. 豁免机制：若预测为裂缝，且落在了我们设定的缓冲带内，则惩罚乘以 alpha (0.1)
        weight_mask = torch.ones_like(bce_loss)
        relax_condition = (clean_target == 0) & (m_relax == 1)
        weight_mask[relax_condition] = self.alpha
        
        # 5. 最终 Loss：结合权重掩码和 255 忽略掩码
        final_loss = bce_loss * weight_mask * valid_mask
        
        # 求平均时只除以有效的像素个数
        return final_loss.sum() / (valid_mask.sum() + 1e-6)
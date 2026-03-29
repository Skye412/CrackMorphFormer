import torch
import torch.nn as nn
import torch.nn.functional as F

class BoundaryRelaxedLoss(nn.Module):
    """
    创新点：边界松弛损失 (Boundary-Relaxed Loss)
    针对 S2DS 数据集中断裂的裂缝标注，给予模型连续预测“豁免权”。
    """
    def __init__(self, alpha=0.1, dilation_kernel=5, ignore_index=255):
        super(BoundaryRelaxedLoss, self).__init__()
        self.alpha = alpha                # 缓冲带内假阳性的惩罚权重 (建议 0.1)
        self.kernel_size = dilation_kernel# 膨胀大小，对应标注断裂的像素距离
        self.ignore_index = ignore_index

    def forward(self, pred_logits, target):
        # target: [B, 1, H, W], 取值 0, 1, 255
        valid_mask = (target != self.ignore_index).float()
        clean_target = target.clone().float()
        clean_target[target == self.ignore_index] = 0.0 
        
        # 形态学膨胀制造容错带 (B, 1, H, W)
        padding = self.kernel_size // 2
        dilated_target = F.max_pool2d(clean_target, kernel_size=self.kernel_size, stride=1, padding=padding)
        m_relax = dilated_target - clean_target
        
        # 基础 BCE
        bce_loss = F.binary_cross_entropy_with_logits(pred_logits, clean_target, reduction='none')
        
        # 如果像素在缓冲带(m_relax==1)且真实标签为背景(clean_target==0)，则惩罚乘以 alpha
        weight_mask = torch.ones_like(bce_loss)
        relax_condition = (clean_target == 0) & (m_relax == 1)
        weight_mask[relax_condition] = self.alpha
        
        final_loss = bce_loss * weight_mask * valid_mask
        return final_loss.sum() / (valid_mask.sum() + 1e-6)
import torch
import torch.nn as nn
from ultralytics.models.yolo.detect import DetectionTrainer
from core.gaussian_loss import v8GaussianDetectionLoss, v8GaussianE2EDetectLoss

def get_detect_head(model):
    """Locates the Detect head module in a YOLO model."""
    m = model.model if hasattr(model, 'model') else model
    
    if hasattr(m, 'model') and isinstance(m.model, nn.Sequential):
        return m.model[-1]
    
    for module in m.modules():
        if type(module).__name__ in ('Detect', 'v10Detect', 'Segment', 'Pose', 'OBB') and hasattr(module, 'cv2'):
            return module
    return None

def adapt_yolo26_for_gaussian(model):
    """
    Adapts a YOLO26 / Ultralytics YOLO model head to output 8 channels per anchor
    (4 for mean prediction mu_hat, 4 for variance prediction Sigma_hat).

    Args:
        model: YOLO PyTorch model instance (e.g. YOLO("yolo26m.pt"))
    """
    detect_head = get_detect_head(model)
    if detect_head is None:
        raise RuntimeError("Could not locate Detect head module in YOLO model.")

    cv2_lists = [detect_head.cv2]
    if hasattr(detect_head, 'one2one_cv2') and detect_head.one2one_cv2 is not None:
        cv2_lists.append(detect_head.one2one_cv2)

    for cv2_list in cv2_lists:
        for i, seq in enumerate(cv2_list):
            if isinstance(seq, nn.Sequential) and len(seq) > 2:
                last_conv = seq[2]
                if isinstance(last_conv, nn.Conv2d):
                    if last_conv.out_channels == 8:
                        continue # Already adapted
                    in_channels = last_conv.in_channels
                    out_channels = 8
                    
                    new_conv = nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=last_conv.kernel_size,
                        stride=last_conv.stride,
                        padding=last_conv.padding,
                        bias=last_conv.bias is not None
                    )
                    
                    # Copy existing weights for mean channels (first 4 channels)
                    with torch.no_grad():
                        num_copy = min(4, last_conv.out_channels)
                        new_conv.weight[:num_copy] = last_conv.weight[:num_copy]
                        # Initialize variance channels (channels 4..7) with small values
                        nn.init.normal_(new_conv.weight[num_copy:], mean=0.0, std=0.01)
                        
                        if last_conv.bias is not None:
                            new_conv.bias[:num_copy] = last_conv.bias[:num_copy]
                            new_conv.bias[num_copy:] = 0.0 # Sigmoid(0.0) = 0.5
                    
                    seq[2] = new_conv

    # Update head attributes
    detect_head.reg_max = 1
    detect_head.no = detect_head.nc + 8

    return model

class GaussianDetectionTrainer(DetectionTrainer):
    """
    Custom DetectionTrainer subclass that returns dual-head v8GaussianE2EDetectLoss.
    """
    def get_loss(self):
        if not hasattr(self, 'criterion') or self.criterion is None:
            self.criterion = v8GaussianE2EDetectLoss(self.model)
        return self.criterion

class GaussianDetectionSingleHeadTrainer(DetectionTrainer):
    """
    Custom DetectionTrainer subclass that returns single-head v8GaussianDetectionLoss.
    """
    def get_loss(self):
        if not hasattr(self, 'criterion') or self.criterion is None:
            self.criterion = v8GaussianDetectionLoss(self.model)
        return self.criterion

class GaussianDetectionTrainerWithL1(DetectionTrainer):
    """
    Custom DetectionTrainer subclass that returns v8GaussianDetectionLossWithL1.
    """
    def get_loss(self):
        if not hasattr(self, 'criterion') or self.criterion is None:
            self.criterion = v8GaussianDetectionLossWithL1(self.model)
        return self.criterion

from core.gaussian_loss import (
    v8GaussianDetectionLoss,
    v8GaussianE2EDetectLoss,
    v8GaussianDetectionLossWithL1,
    v8GaussianE2EDetectLossWithL1,
    v8GaussianCIoUDetectionLoss,
    v8GaussianCIoUE2EDetectLoss,
    v8ImprovedGaussianE2EDetectLoss,
)

class GaussianCIoUDetectionSingleHeadTrainer(DetectionTrainer):
    """
    Custom DetectionTrainer subclass that returns v8GaussianCIoUDetectionLoss (Single-Head Gaussian NLL + CIoU Loss).
    """
    def get_loss(self):
        if not hasattr(self, 'criterion') or self.criterion is None:
            self.criterion = v8GaussianCIoUDetectionLoss(self.model)
        self.criterion.trainer = self
        return self.criterion

class GaussianCIoUE2EDetectionTrainer(DetectionTrainer):
    """
    Custom DetectionTrainer subclass that returns v8GaussianCIoUE2EDetectLoss (Dual-Head E2E Gaussian NLL + CIoU Loss).
    """
    def get_loss(self):
        if not hasattr(self, 'criterion') or self.criterion is None:
            self.criterion = v8GaussianCIoUE2EDetectLoss(self.model)
        self.criterion.trainer = self
        if hasattr(self.criterion, "one2many"):
            self.criterion.one2many.trainer = self
        if hasattr(self.criterion, "one2one"):
            self.criterion.one2one.trainer = self
        return self.criterion

class GaussianImprovedE2EDetectionTrainer(DetectionTrainer):
    """
    Custom DetectionTrainer subclass that returns v8ImprovedGaussianE2EDetectLoss
    (NLL + Variance Regularization + GIoU Loss).
    """
    def get_loss(self):
        if not hasattr(self, 'criterion') or self.criterion is None:
            self.criterion = v8ImprovedGaussianE2EDetectLoss(self.model)
        self.criterion.trainer = self
        if hasattr(self.criterion, "one2many"):
            self.criterion.one2many.trainer = self
        if hasattr(self.criterion, "one2one"):
            self.criterion.one2one.trainer = self
        return self.criterion

def apply_gaussian_inference_criterion(pred_scores: torch.Tensor, pred_sigma: torch.Tensor) -> torch.Tensor:
    """
    Calculates final detection confidence score during inference:
        Score_final = P(Class_c) * (1 - (Sigma_l + Sigma_t + Sigma_r + Sigma_b) / 4)
    """
    avg_sigma = pred_sigma.mean(dim=-1, keepdim=True)
    uncertainty_factor = (1.0 - avg_sigma).clamp(min=0.0, max=1.0)
    return pred_scores * uncertainty_factor

def apply_improved_gaussian_inference_criterion(pred_scores: torch.Tensor, pred_sigma: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    """
    Calculates final detection confidence score during inference using soft exponential attenuation:
        Score_final = P(Class_c) * exp(-gamma * avg_sigma^2)
    """
    avg_sigma = pred_sigma.mean(dim=-1, keepdim=True)
    uncertainty_factor = torch.exp(-gamma * (avg_sigma ** 2))
    return pred_scores * uncertainty_factor

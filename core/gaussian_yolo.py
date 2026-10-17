import types
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils.tal import make_anchors, dist2bbox
from ultralytics.utils.torch_utils import unwrap_model

from core.gaussian_loss import (
    v8GaussianDetectionLoss,
    v8GaussianE2EDetectLoss,
    v8GaussianCIoUDetectionLoss,
    v8GaussianCIoUE2EDetectLoss,
    v8GaussianWithCIoUDetectionLoss,
    v8GaussianWithCIoUE2EDetectLoss,
    v8GaussianGWDDetectionLoss,
    v8GaussianGWDE2EDetectLoss,
    v8GaussianGWDQFLDetectionLoss,
    v8GaussianGWDQFLE2EDetectLoss,
    GaussianCIoUNLLBboxLoss,
    GaussianBboxWithCIoULoss,
    GaussianWassersteinBboxLoss,
)


def get_detect_head(model):
    """Locates the Detect head module in a YOLO model."""
    m = model.model if hasattr(model, 'model') else model
    
    if hasattr(m, 'model') and isinstance(m.model, nn.Sequential):
        return m.model[-1]
    
    for module in m.modules():
        if type(module).__name__ in ('Detect', 'v10Detect', 'Segment', 'Pose', 'OBB') and hasattr(module, 'cv2'):
            return module
    return None


from ultralytics.nn.modules.head import Detect

# Patch Detect._get_decode_boxes to handle Gaussian 8-channel predictions natively
_orig_get_decode_boxes = Detect._get_decode_boxes

def _gaussian_aware_get_decode_boxes(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
    shape = x["feats"][0].shape
    if self.dynamic or self.shape != shape:
        self.anchors, self.strides = (
            a.transpose(0, 1) for a in make_anchors(x["feats"], self.stride, 0.5)
        )
        self.shape = shape
    # If Gaussian head (8 channels: 4 mu + 4 sigma), decode mu with Softplus
    if x["boxes"].shape[1] == 8:
        mu_raw = x["boxes"][:, :4, :]
        mu_pixels = F.softplus(mu_raw)
        dbox = (
            dist2bbox(
                mu_pixels,
                self.anchors.unsqueeze(0),
                xywh=not self.end2end and not self.xyxy,
                dim=1,
            )
            * self.strides
        )
        return dbox
    return _orig_get_decode_boxes(self, x)

Detect._get_decode_boxes = _gaussian_aware_get_decode_boxes


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
                        continue  # Already adapted
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
                            new_conv.bias[num_copy:] = 0.0  # Sigmoid(0.0) = 0.5
                    
                    seq[2] = new_conv

    # Update head attributes
    detect_head.reg_max = 2  # 4 * reg_max = 8 channels (4 for mu, 4 for sigma)
    detect_head.no = detect_head.nc + 8

    return model


from ultralytics.nn.tasks import DetectionModel

_orig_detection_model_init_criterion = DetectionModel.init_criterion

def _gaussian_aware_init_criterion(self):
    detect_head = get_detect_head(self)
    if detect_head is not None and getattr(detect_head, "reg_max", 16) == 2:
        return v8GaussianCIoUE2EDetectLoss(self) if getattr(self, "end2end", False) else v8GaussianCIoUDetectionLoss(self)
    return _orig_detection_model_init_criterion(self)

DetectionModel.init_criterion = _gaussian_aware_init_criterion


class GaussianDetectionTrainer(DetectionTrainer):
    """
    Custom DetectionTrainer subclass that uses dual-head v8GaussianCIoUE2EDetectLoss.
    """
    def get_model(self, cfg: str | None = None, weights: str | None = None, verbose: bool = True):
        model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
        adapt_yolo26_for_gaussian(model)
        return model


class GaussianDetectionSingleHeadTrainer(DetectionTrainer):
    """
    Custom DetectionTrainer subclass that uses single-head v8GaussianDetectionLoss.
    """
    def get_model(self, cfg: str | None = None, weights: str | None = None, verbose: bool = True):
        model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
        adapt_yolo26_for_gaussian(model)
        return model


class GaussianDetectionTrainerWithL1(DetectionTrainer):
    """
    Custom DetectionTrainer subclass that uses v8GaussianDetectionLossWithL1.
    """
    def get_model(self, cfg: str | None = None, weights: str | None = None, verbose: bool = True):
        model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
        adapt_yolo26_for_gaussian(model)
        return model


class GaussianImprovedE2EDetectionTrainer(DetectionTrainer):
    """
    Custom DetectionTrainer subclass that uses v8ImprovedGaussianE2EDetectLoss
    (NLL + Variance Regularization + GIoU Loss).
    """
    def get_model(self, cfg: str | None = None, weights: str | None = None, verbose: bool = True):
        model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
        adapt_yolo26_for_gaussian(model)
        return model


class GaussianWithCIoUDetectionTrainer(DetectionTrainer):
    """
    Custom DetectionTrainer subclass that uses v8GaussianWithCIoUE2EDetectLoss (User Spec Variant).
    """
    def get_model(self, cfg: str | None = None, weights: str | None = None, verbose: bool = True):
        model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
        adapt_yolo26_for_gaussian(model)
        return model


class GaussianGWDDetectionTrainer(DetectionTrainer):
    """
    Custom DetectionTrainer subclass that uses dual-head v8GaussianGWDE2EDetectLoss (Method 3: 2D Gaussian Wasserstein Distance).
    """
    def get_model(self, cfg: str | None = None, weights: str | None = None, verbose: bool = True):
        model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
        adapt_yolo26_for_gaussian(model)
        return model


class GaussianGWDQFLDetectionTrainer(DetectionTrainer):
    """
    Custom DetectionTrainer subclass that uses dual-head v8GaussianGWDQFLE2EDetectLoss (2D GWD + Quality Focal Loss).
    """
    def get_model(self, cfg: str | None = None, weights: str | None = None, verbose: bool = True):
        model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
        adapt_yolo26_for_gaussian(model)
        return model


def apply_improved_gaussian_inference_criterion(pred_scores: torch.Tensor, pred_sigma: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    """
    Calculates final detection confidence score during inference using soft exponential attenuation:
        Score_final = P(Class_c) * exp(-gamma * avg_sigma^2)

    Args:
        pred_scores: Class probabilities / confidence tensor P(Class_c), shape (..., num_classes)
        pred_sigma: Predicted variances Sigma in (0, 1), shape (..., 4) for (l, t, r, b)
        gamma: Temperature scale for exponential attenuation (default 1.0)

    Returns:
        score_final: Confidence scores adjusted by soft exponential uncertainty attenuation.
    """
    avg_sigma = pred_sigma.mean(dim=-1, keepdim=True)  # (..., 1)
    uncertainty_factor = torch.exp(-gamma * (avg_sigma ** 2))
    return pred_scores * uncertainty_factor

import csv
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.utils.loss import v8DetectionLoss, E2ELoss
from ultralytics.utils.tal import TaskAlignedAssigner, dist2bbox, make_anchors
from ultralytics.utils.metrics import bbox_iou

_VARIANCE_LOG_STEP = 0

def log_training_variance_to_csv(sigma_tensor: torch.Tensor, csv_path: str = "training_variance_log.csv"):
    """
    Logs average predicted variance Sigma per batch during training to CSV.
    """
    global _VARIANCE_LOG_STEP
    if not sigma_tensor.numel():
        return
    with torch.no_grad():
        s_l = sigma_tensor[:, 0].mean().item()
        s_t = sigma_tensor[:, 1].mean().item()
        s_r = sigma_tensor[:, 2].mean().item()
        s_b = sigma_tensor[:, 3].mean().item()
        s_avg = sigma_tensor.mean().item()
        s_min = sigma_tensor.min().item()
        s_max = sigma_tensor.max().item()

    _VARIANCE_LOG_STEP += 1
    file_exists = os.path.exists(csv_path)
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['step', 'mean_sigma_l', 'mean_sigma_t', 'mean_sigma_r', 'mean_sigma_b', 'batch_avg_sigma', 'min_sigma', 'max_sigma'])
        writer.writerow([_VARIANCE_LOG_STEP, f"{s_l:.4f}", f"{s_t:.4f}", f"{s_r:.4f}", f"{s_b:.4f}", f"{s_avg:.4f}", f"{s_min:.4f}", f"{s_max:.4f}"])

class GaussianBboxLoss(nn.Module):
    """
    Negative Log-Likelihood Gaussian Bounding Box (LTRB) Loss with Size Compensation Weight & Optional L1 Loss.
    
    Formula from Word Spec:
        L_bbox_Gaussian = sum_{ij} [ I_pos(i, j) * w_scale(i, j) * sum_{theta in LTRB} ( log(Sigma_theta + eps) + (theta_G - mu_theta)^2 / (2 * Sigma_theta^2 + eps) ) ]
        where:
            mu_theta = Softplus(mu_hat) > 0 (normalized distance expectation)
            Sigma_theta = Sigmoid(Sigma_hat) in (0, 1) (predicted variance / uncertainty)
            w_scale = 2 - w_norm_G * h_norm_G (size compensation weight)
            w_norm_G = (x2_G - x1_G) / I_W
            h_norm_G = (y2_G - y1_G) / I_H
    """
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        pred_dist: torch.Tensor,       # (N_pos, 8) or (B, N, 8): 4 for mu_hat, 4 for Sigma_hat
        pred_bboxes: torch.Tensor,     # (B, N, 4) in xyxy pixel coordinates
        anchor_points: torch.Tensor,   # (N, 2) in grid units
        target_bboxes: torch.Tensor,   # (N_pos, 4) in pixel coordinates (x1, y1, x2, y2)
        target_scores: torch.Tensor,   # (B, N, num_classes)
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,         # (B, N) boolean mask for positive samples
        imgsz: torch.Tensor,          # (2,) [H_img, W_img] in pixels
        stride: torch.Tensor,         # (N, 1) stride for each anchor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not fg_mask.any():
            zero = pred_dist.sum() * 0.0
            return zero, zero

        # Handle batch dimensions for fg_mask (B, N)
        b, n = fg_mask.shape
        
        # Expand stride to (B, N, 1) if necessary
        if stride.dim() == 2:
            stride_b = stride.unsqueeze(0).expand(b, n, -1)
        elif stride.dim() == 3:
            stride_b = stride.expand(b, n, -1)
        else:
            stride_b = stride.view(b, n, -1)
        stride_pos = stride_b[fg_mask] # (N_pos, 1)

        # Extract positive predictions
        pred_pos = pred_dist[fg_mask]  # (N_pos, 8)
        pred_mu_raw = pred_pos[:, :4]    # (N_pos, 4)
        pred_sigma_raw = pred_pos[:, 4:] # (N_pos, 4)

        # 1. Predicted expectation mu_theta in pixels (Softplus)
        mu_pixels = F.softplus(pred_mu_raw) * stride_pos # (N_pos, 4) in pixels

        # 2. Predicted variance Sigma_theta in (0, 1) (Sigmoid)
        sigma = torch.sigmoid(pred_sigma_raw) # (N_pos, 4)

        if self.training:
            log_training_variance_to_csv(sigma)

        # 3. Ground Truth normalized distance vector theta_G (LTRB)
        tgt_pos = target_bboxes[fg_mask] # (N_pos, 4) [x1_G, y1_G, x2_G, y2_G]
        
        # Expand anchor_points to (B, N, 2)
        if anchor_points.dim() == 2:
            anchor_b = anchor_points.unsqueeze(0).expand(b, n, -1)
        else:
            anchor_b = anchor_points
        anchors_pos = anchor_b[fg_mask] * stride_pos # (N_pos, 2) anchor coords in pixels

        # Distance from anchor center to GT edges in pixels
        lt_px = anchors_pos - tgt_pos[:, :2]  # l = anc_x - x1_G, t = anc_y - y1_G
        rb_px = tgt_pos[:, 2:] - anchors_pos  # r = x2_G - anc_x, b = y2_G - anc_y
        tgt_dist_px = torch.cat([lt_px, rb_px], dim=-1) # (N_pos, 4) [l_G, t_G, r_G, b_G] in pixels

        # Image width and height for normalization
        img_h, img_w = imgsz[0], imgsz[1]
        scale_factors = torch.tensor([img_w, img_h, img_w, img_h], device=pred_dist.device, dtype=pred_dist.dtype)

        # Normalize predicted distances and target distances by image dimensions
        mu_norm = mu_pixels / scale_factors        # (N_pos, 4)
        tgt_dist_norm = tgt_dist_px / scale_factors # (N_pos, 4)

        # 4. Size compensation weight: w_scale = 2 - w_norm_G * h_norm_G
        w_gt = (tgt_pos[:, 2] - tgt_pos[:, 0]) # x2_G - x1_G
        h_gt = (tgt_pos[:, 3] - tgt_pos[:, 1]) # y2_G - y1_G
        w_norm_g = (w_gt / img_w).clamp(0.0, 1.0)
        h_norm_g = (h_gt / img_h).clamp(0.0, 1.0)
        w_scale = 2.0 - (w_norm_g * h_norm_g) # (N_pos,) in [1.0, 2.0]
        w_scale = w_scale.unsqueeze(-1)       # (N_pos, 1)

        # 5. Negative Log-Likelihood Loss per edge theta:
        diff_sq = (tgt_dist_norm - mu_norm) ** 2
        var_term = 2.0 * (sigma ** 2) + self.eps
        nll = torch.log(sigma + self.eps) + (diff_sq / var_term) # (N_pos, 4)
        nll_sum = nll.sum(dim=-1, keepdim=True) # (N_pos, 1)

        # 6. Auxiliary L1 Loss per edge theta: |tgt_dist_norm - mu_norm|
        l1_dist = torch.abs(tgt_dist_norm - mu_norm).sum(dim=-1, keepdim=True) # (N_pos, 1)

        # 7. Apply Task-Aligned assigner sample weights and size scale weight
        sample_weights = target_scores[fg_mask].sum(-1, keepdim=True) # (N_pos, 1)
        weighted_gaussian = (nll_sum * w_scale * sample_weights).sum()
        weighted_l1 = (l1_dist * sample_weights).sum()

        loss_gaussian = weighted_gaussian / target_scores_sum
        loss_l1 = weighted_l1 / target_scores_sum
        return loss_gaussian, loss_l1


class GaussianBboxWithCIoULoss(nn.Module):
    """
    Gaussian NLL Bbox Loss combined with Complete IoU (CIoU) Loss.
    """
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
        imgsz: torch.Tensor,
        stride: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not fg_mask.any():
            zero = pred_dist.sum() * 0.0
            return zero, zero

        b, n = fg_mask.shape
        if stride.dim() == 2:
            stride_b = stride.unsqueeze(0).expand(b, n, -1)
        elif stride.dim() == 3:
            stride_b = stride.expand(b, n, -1)
        else:
            stride_b = stride.view(b, n, -1)
        stride_pos = stride_b[fg_mask]

        pred_pos = pred_dist[fg_mask]
        pred_mu_raw = pred_pos[:, :4]
        pred_sigma_raw = pred_pos[:, 4:]

        mu_pixels = F.softplus(pred_mu_raw) * stride_pos
        sigma = torch.sigmoid(pred_sigma_raw)

        tgt_pos = target_bboxes[fg_mask]
        if anchor_points.dim() == 2:
            anchor_b = anchor_points.unsqueeze(0).expand(b, n, -1)
        else:
            anchor_b = anchor_points
        anchors_pos = anchor_b[fg_mask] * stride_pos

        lt_px = anchors_pos - tgt_pos[:, :2]
        rb_px = tgt_pos[:, 2:] - anchors_pos
        tgt_dist_px = torch.cat([lt_px, rb_px], dim=-1)

        img_h, img_w = imgsz[0], imgsz[1]
        scale_factors = torch.tensor([img_w, img_h, img_w, img_h], device=pred_dist.device, dtype=pred_dist.dtype)

        mu_norm = mu_pixels / scale_factors
        tgt_dist_norm = tgt_dist_px / scale_factors

        w_gt = (tgt_pos[:, 2] - tgt_pos[:, 0])
        h_gt = (tgt_pos[:, 3] - tgt_pos[:, 1])
        w_norm_g = (w_gt / img_w).clamp(0.0, 1.0)
        h_norm_g = (h_gt / img_h).clamp(0.0, 1.0)
        w_scale = (2.0 - (w_norm_g * h_norm_g)).unsqueeze(-1)

        diff_sq = (tgt_dist_norm - mu_norm) ** 2
        var_term = 2.0 * (sigma ** 2) + self.eps
        nll = torch.log(sigma + self.eps) + (diff_sq / var_term)
        nll_sum = nll.sum(dim=-1, keepdim=True)

        # CIoU Loss calculation
        pred_pos_bboxes = pred_bboxes[fg_mask] * stride_pos
        iou = bbox_iou(pred_pos_bboxes, tgt_pos, CIoU=True, eps=1e-7)
        ciou_loss = (1.0 - iou).view(-1, 1)

        sample_weights = target_scores[fg_mask].sum(-1, keepdim=True)
        weighted_nll = (nll_sum * w_scale * sample_weights).sum()
        weighted_ciou = (ciou_loss * w_scale * sample_weights).sum()

        loss_gaussian = weighted_nll / target_scores_sum
        loss_ciou = weighted_ciou / target_scores_sum
        return loss_gaussian, loss_ciou


class ImprovedGaussianBboxLoss(nn.Module):
    """
    Improved Gaussian Bounding Box Loss combining:
    1. Negative Log-Likelihood (NLL) with Size Compensation Weight w_scale
    2. Variance Regularization Penalty L_var (forces Sigma -> 0 on clean positive samples)
    3. Geometric GIoU Loss L_giou (enforces bounding box shape and overlap alignment)
    """
    def __init__(self, var_weight: float = 0.15, giou_weight: float = 0.5, eps: float = 1e-6):
        super().__init__()
        self.var_weight = var_weight
        self.giou_weight = giou_weight
        self.eps = eps

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
        imgsz: torch.Tensor,
        stride: torch.Tensor,
    ) -> torch.Tensor:
        if not fg_mask.any():
            return pred_dist.sum() * 0.0

        b, n = fg_mask.shape
        if stride.dim() == 2:
            stride_b = stride.unsqueeze(0).expand(b, n, -1)
        elif stride.dim() == 3:
            stride_b = stride.expand(b, n, -1)
        else:
            stride_b = stride.view(b, n, -1)
        stride_pos = stride_b[fg_mask]

        pred_pos = pred_dist[fg_mask]
        pred_mu_raw = pred_pos[:, :4]
        pred_sigma_raw = pred_pos[:, 4:]

        mu_pixels = F.softplus(pred_mu_raw) * stride_pos
        sigma = torch.sigmoid(pred_sigma_raw)

        tgt_pos = target_bboxes[fg_mask]
        if anchor_points.dim() == 2:
            anchor_b = anchor_points.unsqueeze(0).expand(b, n, -1)
        else:
            anchor_b = anchor_points
        anchors_pos = anchor_b[fg_mask] * stride_pos

        lt_px = anchors_pos - tgt_pos[:, :2]
        rb_px = tgt_pos[:, 2:] - anchors_pos
        tgt_dist_px = torch.cat([lt_px, rb_px], dim=-1)

        img_h, img_w = imgsz[0], imgsz[1]
        scale_factors = torch.tensor([img_w, img_h, img_w, img_h], device=pred_dist.device, dtype=pred_dist.dtype)

        mu_norm = mu_pixels / scale_factors
        tgt_dist_norm = tgt_dist_px / scale_factors

        w_gt = (tgt_pos[:, 2] - tgt_pos[:, 0])
        h_gt = (tgt_pos[:, 3] - tgt_pos[:, 1])
        w_norm_g = (w_gt / img_w).clamp(0.0, 1.0)
        h_norm_g = (h_gt / img_h).clamp(0.0, 1.0)
        w_scale = (2.0 - (w_norm_g * h_norm_g)).unsqueeze(-1)

        diff_sq = (tgt_dist_norm - mu_norm) ** 2
        var_term = 2.0 * (sigma ** 2) + self.eps
        nll = torch.log(sigma + self.eps) + (diff_sq / var_term)
        nll_sum = nll.sum(dim=-1, keepdim=True)

        var_penalty = sigma.sum(dim=-1, keepdim=True)

        pred_pos_bboxes = pred_bboxes[fg_mask] * stride_pos
        iou = bbox_iou(pred_pos_bboxes, tgt_pos, GIoU=True, eps=1e-7)
        giou_loss = 1.0 - iou

        sample_weights = target_scores[fg_mask].sum(-1, keepdim=True)
        weighted_loss = (nll_sum * w_scale + self.var_weight * var_penalty + self.giou_weight * giou_loss * w_scale) * sample_weights

        return weighted_loss.sum() / target_scores_sum


def log_per_image_training_variance(
    pred_distri: torch.Tensor,
    fg_mask: torch.Tensor,
    im_files: list | None = None,
    trainer: any = None,
    csv_path: str = "/media/data3/home/truongduy/FL-YOLOv8-Object-Detection/per_image_training_variance_log.csv"
):
    """
    Logs per-image average predicted mean (Mu) and predicted variance (Sigma) for positive anchors
    along with FL round, client epoch, and client name during training to CSV.
    """
    if not fg_mask.any():
        return

    b_size = fg_mask.shape[0]
    file_exists = os.path.exists(csv_path)

    try:
        current_epoch = 1
        current_round = "round1"
        client_name = "client"

        if trainer is not None:
            if hasattr(trainer, "epoch"):
                current_epoch = trainer.epoch + 1
            if hasattr(trainer, "args") and hasattr(trainer.args, "name"):
                run_name = str(trainer.args.name)
                parts = run_name.split("_")
                if len(parts) >= 2:
                    client_name = parts[0]
                    current_round = parts[1]
                elif len(parts) == 1:
                    client_name = parts[0]

        with torch.no_grad():
            rows = []
            for b in range(b_size):
                fg_b = fg_mask[b]
                if not fg_b.any():
                    continue
                
                # 1. Compute Mean Mu (decoded softplus offsets in pixels: Left, Top, Right, Bottom)
                mu_raw = pred_distri[b, fg_b, :4] # (N_pos_b, 4)
                mu = F.softplus(mu_raw)
                m_l = mu[:, 0].mean().item()
                m_t = mu[:, 1].mean().item()
                m_r = mu[:, 2].mean().item()
                m_b = mu[:, 3].mean().item()
                m_w = (mu[:, 0] + mu[:, 2]).mean().item()
                m_h = (mu[:, 1] + mu[:, 3]).mean().item()
                m_avg = mu.mean().item()

                # 2. Compute Mean Sigma (sigmoid variance probabilities: Left, Top, Right, Bottom)
                if pred_distri.shape[-1] >= 8:
                    sig_raw = pred_distri[b, fg_b, 4:8]
                    sig = torch.sigmoid(sig_raw)
                else:
                    sig = torch.ones_like(mu) * 0.5
                s_l = sig[:, 0].mean().item()
                s_t = sig[:, 1].mean().item()
                s_r = sig[:, 2].mean().item()
                s_b = sig[:, 3].mean().item()
                s_avg = sig.mean().item()
                s_min = sig.min().item()
                s_max = sig.max().item()

                img_name = os.path.basename(str(im_files[b])) if (im_files and b < len(im_files)) else f"img_b{b}"
                category = "Clean" if s_avg < 0.15 else ("Medium" if s_avg <= 0.35 else "Noisy")
                
                rows.append([
                    current_round,
                    current_epoch,
                    client_name,
                    img_name,
                    sig.shape[0],
                    f"{m_l:.4f}",
                    f"{m_t:.4f}",
                    f"{m_r:.4f}",
                    f"{m_b:.4f}",
                    f"{m_w:.4f}",
                    f"{m_h:.4f}",
                    f"{m_avg:.4f}",
                    f"{s_l:.4f}",
                    f"{s_t:.4f}",
                    f"{s_r:.4f}",
                    f"{s_b:.4f}",
                    f"{s_avg:.4f}",
                    f"{s_min:.4f}",
                    f"{s_max:.4f}",
                    category
                ])

        if rows:
            with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow([
                        'round', 'epoch', 'client_name', 'image_name', 'num_pos_anchors',
                        'mean_mu_l', 'mean_mu_t', 'mean_mu_r', 'mean_mu_b', 'mean_mu_w', 'mean_mu_h', 'mean_mu_avg',
                        'mean_sigma_l', 'mean_sigma_t', 'mean_sigma_r', 'mean_sigma_b', 'batch_avg_sigma',
                        'min_sigma', 'max_sigma', 'uncertainty_category'
                    ])
                writer.writerows(rows)
    except Exception as err:
        print(f"⚠️ [CSV Logging Error]: {err}", flush=True)


class v8GaussianDetectionLoss(v8DetectionLoss):
    """
    Detection Loss integrating custom Gaussian Bounding Box Loss into Ultralytics pipeline.
    """
    def __init__(self, model, tal_topk: int = 10, tal_topk2: int | None = None, enable_l1: bool = False):
        super().__init__(model, tal_topk=tal_topk, tal_topk2=tal_topk2)
        self.gaussian_bbox_loss = GaussianBboxLoss()
        self.enable_l1 = enable_l1
        self.loss_names = ("box_loss", "cls_loss", "l1_loss") if enable_l1 else ("box_loss", "cls_loss", "dfl_loss")

    def bbox_decode(self, anchor_points: torch.Tensor, pred_dist: torch.Tensor) -> torch.Tensor:
        pred_mu_raw = pred_dist[..., :4]
        mu_pixels = F.softplus(pred_mu_raw)
        return dist2bbox(mu_pixels, anchor_points, xywh=False)

    def get_assigned_targets_and_loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple:
        loss = torch.zeros(3, device=self.device)
        pred_distri = preds["boxes"].permute(0, 2, 1).contiguous()
        pred_scores = preds["scores"].permute(0, 2, 1).contiguous()
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]

        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)
        
        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1.0)

        # 1. Classification Loss (BCE)
        bce_loss = self.bce(pred_scores, target_scores.to(dtype))
        if self.class_weights is not None:
            bce_loss *= self.class_weights
        loss[1] = bce_loss.sum() / target_scores_sum

        # 2. Bbox Loss
        if fg_mask.sum():
            loss_g, loss_l1 = self.gaussian_bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes,
                target_scores,
                target_scores_sum,
                fg_mask,
                imgsz,
                stride_tensor,
            )
            loss[0] = loss_g
            loss[2] = loss_l1 if self.enable_l1 else torch.tensor(0.0, device=self.device)

            if pred_distri.requires_grad or getattr(self, "training", True):
                log_per_image_training_variance(pred_distri, fg_mask, batch.get("im_file"), trainer=getattr(self, "trainer", None))

        return loss * batch_size, torch.cat((fg_mask.unsqueeze(-1), target_gt_idx.unsqueeze(-1)), 2)


class v8GaussianCIoUDetectionLoss(v8DetectionLoss):
    """
    Detection Loss combining Gaussian NLL Loss + CIoU Loss for single-head detection.
    """
    def __init__(self, model, tal_topk: int = 10, tal_topk2: int | None = None, ciou_weight: float = 0.5):
        super().__init__(model, tal_topk=tal_topk, tal_topk2=tal_topk2)
        self.gaussian_ciou_bbox_loss = GaussianBboxWithCIoULoss()
        self.ciou_weight = ciou_weight
        self.loss_names = ("box_loss", "cls_loss", "ciou_loss")

    def bbox_decode(self, anchor_points: torch.Tensor, pred_dist: torch.Tensor) -> torch.Tensor:
        pred_mu_raw = pred_dist[..., :4]
        mu_pixels = F.softplus(pred_mu_raw)
        return dist2bbox(mu_pixels, anchor_points, xywh=False)

    def get_assigned_targets_and_loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple:
        loss = torch.zeros(3, device=self.device)
        pred_distri = preds["boxes"].permute(0, 2, 1).contiguous()
        pred_scores = preds["scores"].permute(0, 2, 1).contiguous()
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]

        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)
        
        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1.0)

        # 1. Classification Loss (BCE)
        bce_loss = self.bce(pred_scores, target_scores.to(dtype))
        if self.class_weights is not None:
            bce_loss *= self.class_weights
        loss[1] = bce_loss.sum() / target_scores_sum

        # 2. Bbox Loss (Gaussian NLL + CIoU Loss)
        if fg_mask.sum():
            loss_g, loss_ciou = self.gaussian_ciou_bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes,
                target_scores,
                target_scores_sum,
                fg_mask,
                imgsz,
                stride_tensor,
            )
            loss[0] = loss_g    # Receives self.hyp.box (7.5)
            loss[2] = loss_ciou # Receives self.hyp.dfl (1.5)

            if pred_distri.requires_grad or getattr(self, "training", True):
                log_per_image_training_variance(pred_distri, fg_mask, batch.get("im_file"), trainer=getattr(self, "trainer", None))

        return loss * batch_size, torch.cat((fg_mask.unsqueeze(-1), target_gt_idx.unsqueeze(-1)), 2)


class v8ImprovedGaussianDetectionLoss(v8DetectionLoss):
    """
    Improved Detection Loss integrating ImprovedGaussianBboxLoss (NLL + Variance Penalty + GIoU Loss).
    """
    def __init__(self, model, tal_topk: int = 10, tal_topk2: int | None = None):
        super().__init__(model, tal_topk=tal_topk, tal_topk2=tal_topk2)
        self.gaussian_bbox_loss = ImprovedGaussianBboxLoss()
        self.loss_names = ("box_loss", "cls_loss", "dfl_loss")

    def bbox_decode(self, anchor_points: torch.Tensor, pred_dist: torch.Tensor) -> torch.Tensor:
        pred_mu_raw = pred_dist[..., :4]
        mu_pixels = F.softplus(pred_mu_raw)
        return dist2bbox(mu_pixels, anchor_points, xywh=False)

    def get_assigned_targets_and_loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple:
        loss = torch.zeros(3, device=self.device)
        pred_distri = preds["boxes"].permute(0, 2, 1).contiguous()
        pred_scores = preds["scores"].permute(0, 2, 1).contiguous()
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]

        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)
        
        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1.0)

        # 1. Classification Loss (BCE)
        bce_loss = self.bce(pred_scores, target_scores.to(dtype))
        if self.class_weights is not None:
            bce_loss *= self.class_weights
        loss[1] = bce_loss.sum() / target_scores_sum

        # 2. Improved Bbox Loss (NLL + Variance Penalty + GIoU)
        if fg_mask.sum():
            loss[0] = self.gaussian_bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes,
                target_scores,
                target_scores_sum,
                fg_mask,
                imgsz,
                stride_tensor,
            )
            loss[2] = torch.tensor(0.0, device=self.device)

        return loss * batch_size, torch.cat((fg_mask.unsqueeze(-1), target_gt_idx.unsqueeze(-1)), 2)


class v8GaussianDetectionLossWithL1(v8GaussianDetectionLoss):
    """v8GaussianDetectionLoss with L1 loss enabled."""
    def __init__(self, model, tal_topk: int = 10, tal_topk2: int | None = None):
        super().__init__(model, tal_topk=tal_topk, tal_topk2=tal_topk2, enable_l1=True)


class v8GaussianE2EDetectLoss(E2ELoss):
    """
    Dual-Head End-to-End Loss (One-to-Many + One-to-One) using Custom Gaussian Bbox Loss
    with dynamic decay schedule (o2m decay 0.8 -> 0.1, o2o growth 0.2 -> 0.9).
    """
    def __init__(self, model, enable_l1: bool = False):
        loss_fn = (lambda m, tal_topk=10, tal_topk2=None: v8GaussianDetectionLoss(m, tal_topk=tal_topk, tal_topk2=tal_topk2, enable_l1=enable_l1))
        super().__init__(model, loss_fn=loss_fn)


class v8GaussianCIoUE2EDetectLoss(E2ELoss):
    """
    Dual-Head End-to-End Gaussian Loss + CIoU Loss (One-to-Many + One-to-One).
    """
    def __init__(self, model):
        loss_fn = (lambda m, tal_topk=10, tal_topk2=None: v8GaussianCIoUDetectionLoss(m, tal_topk=tal_topk, tal_topk2=tal_topk2))
        super().__init__(model, loss_fn=loss_fn)


class v8ImprovedGaussianE2EDetectLoss(E2ELoss):
    """
    Dual-Head End-to-End Improved Gaussian Loss (NLL + Variance Regularization + GIoU Loss).
    """
    def __init__(self, model):
        loss_fn = (lambda m, tal_topk=10, tal_topk2=None: v8ImprovedGaussianDetectionLoss(m, tal_topk=tal_topk, tal_topk2=tal_topk2))
        super().__init__(model, loss_fn=loss_fn)


class v8GaussianE2EDetectLossWithL1(v8GaussianE2EDetectLoss):
    """Dual-Head End-to-End Gaussian Loss with L1 loss enabled."""
    def __init__(self, model):
        super().__init__(model, enable_l1=True)

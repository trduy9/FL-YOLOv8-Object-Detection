import os
import csv
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.utils.loss import v8DetectionLoss, E2ELoss
from ultralytics.utils.tal import make_anchors, dist2bbox
from ultralytics.utils.metrics import bbox_iou

_ROUND_IMAGE_STATS = {}


def _parse_client_round(save_name: str) -> tuple[str, int]:
    try:
        parts = save_name.split("_")
        client_name = parts[0]
        round_str = parts[1]
        round_num = int(round_str.replace("round", ""))
        return client_name, round_num
    except Exception:
        return "client1", 1


def log_per_image_training_variance(
    pred_distri: torch.Tensor,
    fg_mask: torch.Tensor,
    im_files: list | tuple | None = None,
    trainer=None,
):
    """
    Accumulates predicted mean (Mu) and predicted variance (Sigma) per image during training.
    Stats are stored in the module-level _ROUND_IMAGE_STATS dict.
    Call flush_variance_log() at the end of training to write accumulated stats to CSV.
    """
    if pred_distri is None or not fg_mask.any():
        return

    # Only log images from training set, skip validation set
    if im_files and len(im_files) > 0 and im_files[0] is not None:
        first_file = str(im_files[0])
        if os.sep + "val" in first_file or os.sep + "valid" in first_file:
            return

    try:
        current_round_str = "round1"
        client_name = "client1"
        epoch = -1

        if trainer is not None:
            run_name = str(getattr(getattr(trainer, "args", None), "name", "client1_round1"))
            parsed_client, round_num = _parse_client_round(run_name)
            client_name = parsed_client
            current_round_str = f"round{round_num}"
            epoch = int(getattr(trainer, "epoch", -1))

        with torch.no_grad():
            for b in range(fg_mask.shape[0]):
                fg_b = fg_mask[b]
                if not fg_b.any():
                    continue

                if im_files and b < len(im_files) and im_files[b] is not None:
                    img_path = str(im_files[b])
                    if os.sep + "val" in img_path or os.sep + "valid" in img_path:
                        continue
                    img_name = os.path.basename(img_path)
                else:
                    img_name = f"img_sample_{b}"

                # 1. Compute Mean Mu (decoded softplus offsets)
                mu_raw = pred_distri[b, fg_b, :4]
                mu = F.softplus(mu_raw)
                m_w = (mu[:, 0] + mu[:, 2]).mean().item()
                m_h = (mu[:, 1] + mu[:, 3]).mean().item()
                m_avg = mu.mean().item()

                # 2. Compute Mean Sigma (Kendall & Gal log-variance: s = ln(sigma^2) -> sigma = exp(0.5 * s))
                if pred_distri.shape[-1] >= 8:
                    sig_raw = pred_distri[b, fg_b, 4:8]
                    s = sig_raw.clamp(-10.0, 10.0)
                    sig = torch.exp(0.5 * s)
                else:
                    sig = torch.ones_like(mu) * 0.5

                s_l = sig[:, 0].mean().item()
                s_t = sig[:, 1].mean().item()
                s_r = sig[:, 2].mean().item()
                s_b = sig[:, 3].mean().item()
                s_avg = sig.mean().item()
                s_min = sig.min().item()
                s_max = sig.max().item()

                key = (current_round_str, client_name, img_name)
                if key not in _ROUND_IMAGE_STATS:
                    _ROUND_IMAGE_STATS[key] = {
                        'count': 0, 'n_anchors': sig.shape[0],
                        'last_epoch': epoch,
                        'sum_mw': 0.0, 'sum_mh': 0.0, 'sum_mavg': 0.0,
                        'sum_sl': 0.0, 'sum_st': 0.0, 'sum_sr': 0.0, 'sum_sb': 0.0, 'sum_savg': 0.0,
                        'min_s': s_min, 'max_s': s_max
                    }

                st = _ROUND_IMAGE_STATS[key]
                st['count'] += 1
                st['last_epoch'] = max(st.get('last_epoch', -1), epoch)
                st['sum_mw'] += m_w
                st['sum_mh'] += m_h
                st['sum_mavg'] += m_avg
                st['sum_sl'] += s_l
                st['sum_st'] += s_t
                st['sum_sr'] += s_r
                st['sum_sb'] += s_b
                st['sum_savg'] += s_avg
                st['min_s'] = min(st['min_s'], s_min)
                st['max_s'] = max(st['max_s'], s_max)

    except Exception as err:
        print(f"⚠️ [Variance Accumulation Error]: {err}", flush=True)


def flush_variance_log(
    client_name: str = "client1",
    round_num: int = 1,
    last_epoch: int = -1,
    save_dir: str | None = None,
    project_root: str | None = None
):
    """
    Writes accumulated round-averaged Mu/Sigma stats per image to CSV, then clears the buffer.
    Called once at the end of each client's training round.

    - Project root CSV (per_round_uncertainty_{client_name}.csv): APPEND mode — preserves all rounds
    - Client run dir CSV (per_image_variance_log.csv): WRITE mode — fresh per run
    """
    global _ROUND_IMAGE_STATS

    if not _ROUND_IMAGE_STATS:
        print(f"ℹ️ [Variance Log] No stats to flush for {client_name} round{round_num}", flush=True)
        return

    fieldnames = [
        'round', 'client_name', 'epoch', 'image_name', 'num_pos_anchors',
        'round_avg_mu_w', 'round_avg_mu_h', 'round_avg_mu',
        'round_avg_sigma_l', 'round_avg_sigma_t', 'round_avg_sigma_r', 'round_avg_sigma_b', 'round_avg_sigma',
        'min_sigma', 'max_sigma', 'uncertainty_category'
    ]

    round_str = f"round{round_num}"
    rows = []

    for (r, c_name, img_n), st in _ROUND_IMAGE_STATS.items():
        cnt = st['count']
        if cnt == 0:
            continue
        avg_mw = st['sum_mw'] / cnt
        avg_mh = st['sum_mh'] / cnt
        avg_mavg = st['sum_mavg'] / cnt
        avg_sl = st['sum_sl'] / cnt
        avg_st_val = st['sum_st'] / cnt
        avg_sr = st['sum_sr'] / cnt
        avg_sb = st['sum_sb'] / cnt
        avg_savg = st['sum_savg'] / cnt
        epoch = st.get('last_epoch', last_epoch)
        if epoch < 0:
            epoch = last_epoch
        category = "Clean" if avg_savg < 0.15 else ("Medium" if avg_savg <= 0.35 else "Noisy")

        rows.append([
            round_str, c_name, epoch, img_n, st['n_anchors'],
            f"{avg_mw:.4f}", f"{avg_mh:.4f}", f"{avg_mavg:.4f}",
            f"{avg_sl:.4f}", f"{avg_st_val:.4f}", f"{avg_sr:.4f}", f"{avg_sb:.4f}", f"{avg_savg:.4f}",
            f"{st['min_s']:.4f}", f"{st['max_s']:.4f}", category
        ])

    rows.sort(key=lambda x: (x[0], x[3]))

    if not rows:
        _ROUND_IMAGE_STATS.clear()
        return

    try:
        # 1. Project Root CSV — APPEND mode (accumulates across all rounds)
        if project_root is not None:
            root_csv = os.path.join(project_root, f"per_round_uncertainty_{client_name}.csv")
            write_header = not os.path.exists(root_csv) or os.path.getsize(root_csv) == 0
            with open(root_csv, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(fieldnames)
                writer.writerows(rows)

        # 2. Client Run Dir CSV — WRITE mode (fresh per run)
        if save_dir is not None:
            os.makedirs(str(save_dir), exist_ok=True)
            local_csv = os.path.join(str(save_dir), "per_image_variance_log.csv")
            with open(local_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(fieldnames)
                writer.writerows(rows)

        print(f"📊 [Variance Log FLUSHED] Client: {client_name}, Round: {round_str}, "
              f"Epoch: {last_epoch}, Images: {len(rows)}", flush=True)

    except Exception as err:
        print(f"⚠️ [Variance Log Flush Error]: {err}", flush=True)

    _ROUND_IMAGE_STATS.clear()


class GaussianCIoUNLLBboxLoss(nn.Module):
    """
    Hybrid Bounding Box Loss combining:
    1. Complete IoU (CIoU) Loss on decoded bounding boxes
    2. Negative Log-Likelihood (NLL) Loss with Size Compensation Weight w_scale
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

        # 2. Predicted log-variance s = ln(sigma^2) in [-10, 10] (Kendall & Gal)
        s = pred_sigma_raw.clamp(-10.0, 10.0)
        sigma = torch.exp(0.5 * s) # (N_pos, 4)

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

        # 5. Negative Log-Likelihood Loss per edge theta (Kendall & Gal):
        # L_NLL = 0.5 * exp(-s) * (tgt - mu)^2 + 0.5 * s
        diff_sq = (tgt_dist_norm - mu_norm) ** 2
        inv_var = torch.exp(-s)
        nll = 0.5 * inv_var * diff_sq + 0.5 * s # (N_pos, 4)
        nll_sum = nll.sum(dim=-1, keepdim=True) # (N_pos, 1)

        # 6. Complete IoU (CIoU) Loss on predicted boxes:
        pred_pos_bboxes = pred_bboxes[fg_mask] * stride_pos # (N_pos, 4) in pixels xyxy
        ciou = bbox_iou(pred_pos_bboxes, tgt_pos, xywh=False, CIoU=True, eps=1e-7)
        ciou_loss = 1.0 - ciou # (N_pos, 1) in [0, 2]

        # 7. Apply Task-Aligned assigner sample weights and size scale weight
        sample_weights = target_scores[fg_mask].sum(-1, keepdim=True) # (N_pos, 1)
        weighted_ciou = (ciou_loss * sample_weights).sum()
        weighted_nll = (nll_sum * w_scale * sample_weights).sum()

        loss_ciou = weighted_ciou / target_scores_sum
        loss_nll = weighted_nll / target_scores_sum
        return loss_ciou, loss_nll


# Alias for backward compatibility
GaussianBboxLoss = GaussianCIoUNLLBboxLoss


class GaussianBboxWithCIoULoss(nn.Module):
    """
    Gaussian NLL Bbox Loss combined with Complete IoU (CIoU) Loss (User Spec Variant).
    Both NLL and CIoU are weighted by size compensation weight w_scale.
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
        # Kendall & Gal log-variance parameterization: s = ln(sigma^2)
        s = pred_sigma_raw.clamp(-10.0, 10.0)
        sigma = torch.exp(0.5 * s)

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
        inv_var = torch.exp(-s)
        nll = 0.5 * inv_var * diff_sq + 0.5 * s
        nll_sum = nll.sum(dim=-1, keepdim=True)

        # CIoU Loss calculation
        pred_pos_bboxes = pred_bboxes[fg_mask] * stride_pos
        iou = bbox_iou(pred_pos_bboxes, tgt_pos, xywh=False, CIoU=True, eps=1e-7)
        ciou_loss = (1.0 - iou).view(-1, 1)

        sample_weights = target_scores[fg_mask].sum(-1, keepdim=True)
        weighted_nll = (nll_sum * w_scale * sample_weights).sum()
        weighted_ciou = (ciou_loss * w_scale * sample_weights).sum()

        loss_gaussian = weighted_nll / target_scores_sum
        loss_ciou = weighted_ciou / target_scores_sum
        return loss_gaussian, loss_ciou


class v8GaussianWithCIoUDetectionLoss(v8DetectionLoss):
    """
    Detection Loss wrapping GaussianBboxWithCIoULoss with automatic per-image variance & mu logging.
    """
    def __init__(self, model, tal_topk: int = 10, tal_topk2: int | None = None):
        super().__init__(model, tal_topk=tal_topk, tal_topk2=tal_topk2)
        self.gaussian_bbox_loss = GaussianBboxWithCIoULoss()
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

        # 2. Gaussian NLL + CIoU Loss
        if fg_mask.sum():
            loss_gaussian, loss_ciou = self.gaussian_bbox_loss(
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
            loss[0] = loss_ciou       # CIoU loss for box
            loss[2] = loss_gaussian   # Gaussian NLL loss for uncertainty / dfl slot

            # Log per-image variance and mu statistics
            im_files = batch.get("im_file") or batch.get("im_files") or batch.get("img_files")
            log_per_image_training_variance(pred_distri, fg_mask, im_files, trainer=getattr(self, "trainer", None))

        box_gain = float(getattr(self.hyp, "box", 8.0) if getattr(self.hyp, "box", None) is not None else 8.0)
        cls_gain = float(getattr(self.hyp, "cls", 0.5) if getattr(self.hyp, "cls", None) is not None else 0.5)
        dfl_gain = float(getattr(self.hyp, "dfl", 0.5) if getattr(self.hyp, "dfl", None) is not None else 0.5)

        loss[0] *= box_gain  # CIoU gain (8.0)
        loss[1] *= cls_gain  # BCE gain (0.5)
        loss[2] *= dfl_gain  # Gaussian NLL gain (0.5)

        return (
            (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor),
            loss,
            dict(zip(self.loss_names, loss.detach())),
        )


class v8GaussianWithCIoUE2EDetectLoss(E2ELoss):
    """
    Dual-Head End-to-End Detect Loss using GaussianBboxWithCIoULoss (User Spec Variant).
    """
    def __init__(self, model):
        loss_fn = (lambda m, tal_topk=10, tal_topk2=None: v8GaussianWithCIoUDetectionLoss(m, tal_topk=tal_topk, tal_topk2=tal_topk2))
        super().__init__(model, loss_fn=loss_fn)
        self.one2one = v8GaussianWithCIoUDetectionLoss(model, tal_topk=1)


class v8GaussianDetectionLoss(v8DetectionLoss):
    """
    Detection Loss integrating CIoU (7.5) + Gaussian NLL (1.5) + BCE (0.5) into Ultralytics pipeline.
    """
    def __init__(self, model, tal_topk: int = 10, tal_topk2: int | None = None, enable_l1: bool = False):
        super().__init__(model, tal_topk=tal_topk, tal_topk2=tal_topk2)
        self.gaussian_bbox_loss = GaussianCIoUNLLBboxLoss()
        self.enable_l1 = enable_l1
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

        # 2. Hybrid Bbox Loss: CIoU (weight 7.5) + Gaussian NLL (weight 1.5)
        if fg_mask.sum():
            loss_ciou, loss_nll = self.gaussian_bbox_loss(
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
            loss[0] = loss_ciou
            loss[2] = loss_nll

            # Accumulate per-image variance/mu stats for CSV logging (TRAIN only via path filter)
            im_files = batch.get("im_file") or batch.get("im_files") or batch.get("img_files")
            log_per_image_training_variance(pred_distri, fg_mask, im_files, trainer=getattr(self, "trainer", None))

        box_gain = 8.0  # 8.0 * CIoU loss
        cls_gain = float(getattr(self.hyp, "cls", 0.5) if getattr(self.hyp, "cls", None) is not None else 0.5)
        dfl_gain = 1.0  # 0.5 * Gaussian NLL loss

        loss[0] *= box_gain  # 8.0 * CIoU loss
        loss[1] *= cls_gain  # 0.5 * BCE loss
        loss[2] *= dfl_gain  # 0.5 * Gaussian NLL loss

        return (
            (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor),
            loss,
            dict(zip(self.loss_names, loss.detach())),
        )


# Explicit naming
v8GaussianCIoUDetectionLoss = v8GaussianDetectionLoss


class v8GaussianE2EDetectLoss(E2ELoss):
    """
    Dual-Head End-to-End Loss (One-to-Many + One-to-One) using Hybrid CIoU (8.0) + NLL (0.5) + BCE (0.5).
    """
    def __init__(self, model, enable_l1: bool = False):
        loss_fn = (lambda m, tal_topk=10, tal_topk2=None: v8GaussianDetectionLoss(m, tal_topk=tal_topk, tal_topk2=tal_topk2, enable_l1=enable_l1))
        super().__init__(model, loss_fn=loss_fn)


# Explicit naming for Hybrid CIoU + NLL + BCE E2E Loss
v8GaussianCIoUE2EDetectLoss = v8GaussianE2EDetectLoss


# =====================================================================
# METHOD 3: 2D Gaussian Wasserstein Distance (GWD) Bounding Box Loss
# =====================================================================

class GaussianWassersteinBboxLoss(nn.Module):
    """
    2D Gaussian Wasserstein Distance (GWD) Bounding Box Loss with Uncertainty Regularization.
    
    Models predicted and ground-truth bounding boxes as 2D Gaussian distributions:
        N_p = N(mu_p, Sigma_p)
        N_g = N(mu_g, Sigma_g)
    where:
        mu_p = (c_xp, c_yp) (center coordinates)
        Sigma_p = diag(sigma_xp^2, sigma_yp^2)
        sigma_xp^2 = (w_p / 2)^2 + sigma_unc_x^2
        sigma_yp^2 = (h_p / 2)^2 + sigma_unc_y^2
        
    Wasserstein distance W_2^2 has exact closed-form analytical solution:
        W_2^2 = ||mu_p - mu_g||_2^2 + (sigma_xp - sigma_xg)^2 + (sigma_yp - sigma_yg)^2
        
    Normalized GWD Loss:
        L_GWD = 1 - exp(-sqrt(W_2^2 + eps) / C)
    """
    def __init__(self, c_scale: float = 0.5, eps: float = 1e-6):
        super().__init__()
        self.c_scale = c_scale
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

        # Log-variance parameterization (Kendall & Gal) for uncertainty
        s = pred_sigma_raw.clamp(-10.0, 10.0)
        sigma_unc = torch.exp(0.5 * s)  # (N_pos, 4) in (0, inf)

        # Image size normalization
        img_h, img_w = imgsz[0], imgsz[1]
        scale_xyxy = torch.tensor([img_w, img_h, img_w, img_h], device=pred_dist.device, dtype=pred_dist.dtype)

        # 1. Predicted Box in normalized coordinates [0, 1]
        pred_pos_bboxes_px = pred_bboxes[fg_mask] * stride_pos  # (N_pos, 4) in pixels [x1, y1, x2, y2]
        pred_norm = pred_pos_bboxes_px / scale_xyxy

        # 2. Target Box in normalized coordinates [0, 1]
        tgt_pos = target_bboxes[fg_mask]  # (N_pos, 4) [x1, y1, x2, y2] in pixels
        tgt_norm = tgt_pos / scale_xyxy

        # Centers: mu = [(x1 + x2)/2, (y1 + y2)/2]
        c_xp = (pred_norm[:, 0] + pred_norm[:, 2]) * 0.5
        c_yp = (pred_norm[:, 1] + pred_norm[:, 3]) * 0.5
        c_xg = (tgt_norm[:, 0] + tgt_norm[:, 2]) * 0.5
        c_yg = (tgt_norm[:, 1] + tgt_norm[:, 3]) * 0.5

        # Half-lengths (radii) of bounding boxes: r = (x2 - x1)/2
        r_xp = torch.clamp((pred_norm[:, 2] - pred_norm[:, 0]) * 0.5, min=1e-4)
        r_yp = torch.clamp((pred_norm[:, 3] - pred_norm[:, 1]) * 0.5, min=1e-4)
        r_xg = torch.clamp((tgt_norm[:, 2] - tgt_norm[:, 0]) * 0.5, min=1e-4)
        r_yg = torch.clamp((tgt_norm[:, 3] - tgt_norm[:, 1]) * 0.5, min=1e-4)

        # Incorporate predicted uncertainty into spatial distribution variance
        unc_x = (sigma_unc[:, 0] + sigma_unc[:, 2]) * 0.5
        unc_y = (sigma_unc[:, 1] + sigma_unc[:, 3]) * 0.5

        sigma_xp = torch.sqrt(r_xp ** 2 + unc_x ** 2 + self.eps)
        sigma_yp = torch.sqrt(r_yp ** 2 + unc_y ** 2 + self.eps)
        sigma_xg = r_xg
        sigma_yg = r_yg

        # 3. Squared 2-Wasserstein Distance W_2^2:
        dist_center_sq = (c_xp - c_xg) ** 2 + (c_yp - c_yg) ** 2
        dist_scale_sq = (sigma_xp - sigma_xg) ** 2 + (sigma_yp - sigma_yg) ** 2
        w2_sq = dist_center_sq + dist_scale_sq

        # Normalized Gaussian Wasserstein Distance (NWD) loss in [0, 1]
        w2 = torch.sqrt(torch.clamp(w2_sq, min=self.eps))
        loss_gwd = 1.0 - torch.exp(-w2 / self.c_scale)  # (N_pos,) in [0, 1]

        # 4. Uncertainty Regularization Loss (Kendall & Gal NLL on edge distances)
        mu_pixels = F.softplus(pred_mu_raw) * stride_pos
        scale_ltrb = torch.tensor([img_w, img_h, img_w, img_h], device=pred_dist.device, dtype=pred_dist.dtype)
        mu_norm = mu_pixels / scale_ltrb

        if anchor_points.dim() == 2:
            anchor_b = anchor_points.unsqueeze(0).expand(b, n, -1)
        else:
            anchor_b = anchor_points
        anchors_pos = anchor_b[fg_mask] * stride_pos
        lt_px = anchors_pos - tgt_pos[:, :2]
        rb_px = tgt_pos[:, 2:] - anchors_pos
        tgt_dist_norm = torch.cat([lt_px, rb_px], dim=-1) / scale_ltrb

        diff_sq = (tgt_dist_norm - mu_norm) ** 2
        nll_unc = 0.5 * torch.exp(-s) * diff_sq + 0.5 * s
        loss_unc = nll_unc.sum(dim=-1, keepdim=True)  # (N_pos, 1)

        # 5. Size compensation weight w_scale
        w_gt = (tgt_pos[:, 2] - tgt_pos[:, 0])
        h_gt = (tgt_pos[:, 3] - tgt_pos[:, 1])
        w_norm_g = (w_gt / img_w).clamp(0.0, 1.0)
        h_norm_g = (h_gt / img_h).clamp(0.0, 1.0)
        w_scale = (2.0 - (w_norm_g * h_norm_g)).unsqueeze(-1)

        sample_weights = target_scores[fg_mask].sum(-1, keepdim=True)
        weighted_gwd = (loss_gwd.unsqueeze(-1) * w_scale * sample_weights).sum()
        weighted_unc = (loss_unc * sample_weights).sum()

        loss_gwd_mean = weighted_gwd / target_scores_sum
        loss_unc_mean = weighted_unc / target_scores_sum
        return loss_gwd_mean, loss_unc_mean


class v8GaussianGWDDetectionLoss(v8DetectionLoss):
    """
    Detection Loss using 2D Gaussian Wasserstein Distance (GWD) + Uncertainty Regularization (Method 3).
    """
    def __init__(self, model, tal_topk: int = 10, tal_topk2: int | None = None, c_scale: float = 0.5):
        super().__init__(model, tal_topk=tal_topk, tal_topk2=tal_topk2)
        self.gwd_bbox_loss = GaussianWassersteinBboxLoss(c_scale=c_scale)
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

        # 2. 2D Gaussian Wasserstein Distance Loss
        if fg_mask.sum():
            loss_gwd, loss_unc = self.gwd_bbox_loss(
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
            loss[0] = loss_gwd  # GWD loss in [0, 1] for box_loss
            loss[2] = loss_unc  # Uncertainty regularization for dfl_loss

            # Log per-image variance and mu statistics
            im_files = batch.get("im_file") or batch.get("im_files") or batch.get("img_files")
            log_per_image_training_variance(pred_distri, fg_mask, im_files, trainer=getattr(self, "trainer", None))

        box_gain = float(getattr(self.hyp, "box", 8.0) if getattr(self.hyp, "box", None) is not None else 8.0)
        cls_gain = float(getattr(self.hyp, "cls", 0.5) if getattr(self.hyp, "cls", None) is not None else 0.5)
        dfl_gain = float(getattr(self.hyp, "dfl", 0.5) if getattr(self.hyp, "dfl", None) is not None else 0.5)

        loss[0] *= box_gain  # GWD gain (e.g. 8.0)
        loss[1] *= cls_gain  # BCE gain (0.5)
        loss[2] *= dfl_gain  # Uncertainty regularization gain (0.5)

        return (
            (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor),
            loss,
            dict(zip(self.loss_names, loss.detach())),
        )


class v8GaussianGWDE2EDetectLoss(E2ELoss):
    """
    Dual-Head End-to-End Detect Loss using 2D Gaussian Wasserstein Distance (GWD) (Method 3).
    """
    def __init__(self, model, c_scale: float = 0.5):
        loss_fn = (lambda m, tal_topk=10, tal_topk2=None: v8GaussianGWDDetectionLoss(m, tal_topk=tal_topk, tal_topk2=tal_topk2, c_scale=c_scale))
        super().__init__(model, loss_fn=loss_fn)
        self.one2one = v8GaussianGWDDetectionLoss(model, tal_topk=1, c_scale=c_scale)


# =====================================================================
# METHOD OPTIMAL: 2D GWD + Uncertainty-Guided Quality Focal Loss (QFL)
# =====================================================================

class QualityFocalLoss(nn.Module):
    """
    Quality Focal Loss (QFL) for dense object detection:
        L_QFL = |y - p|^gamma * BCEWithLogits(z, y)
    where y is the soft quality target in [0, 1] and p = sigmoid(z).
    """
    def __init__(self, gamma: float = 1.5):
        super().__init__()
        self.gamma = gamma

    def forward(self, pred_logits: torch.Tensor, target_scores: torch.Tensor) -> torch.Tensor:
        pred_sigmoid = pred_logits.sigmoid()
        modulating_factor = torch.abs(target_scores - pred_sigmoid).pow(self.gamma)
        bce = F.binary_cross_entropy_with_logits(pred_logits, target_scores, reduction='none')
        return modulating_factor * bce


class v8GaussianGWDQFLDetectionLoss(v8DetectionLoss):
    """
    Detection Loss using 2D Gaussian Wasserstein Distance (GWD) + Uncertainty-Guided Quality Focal Loss (QFL).
    """
    def __init__(self, model, tal_topk: int = 10, tal_topk2: int | None = None, c_scale: float = 0.5, qfl_gamma: float = 1.5):
        super().__init__(model, tal_topk=tal_topk, tal_topk2=tal_topk2)
        self.gwd_bbox_loss = GaussianWassersteinBboxLoss(c_scale=c_scale)
        self.qfl = QualityFocalLoss(gamma=qfl_gamma)
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

        # 1. Quality Focal Loss (QFL) with Uncertainty Modulation
        target_quality_scores = target_scores.clone().to(dtype)
        if fg_mask.sum() and pred_distri.shape[-1] >= 8:
            s_raw = pred_distri[:, :, 4:8][fg_mask].clamp(-10.0, 10.0)
            sigma_unc = torch.exp(0.5 * s_raw)
            avg_sigma = sigma_unc.mean(dim=-1, keepdim=True)
            unc_factor = torch.exp(-avg_sigma).clamp(0.0, 1.0)
            target_quality_scores[fg_mask] = target_quality_scores[fg_mask] * unc_factor

        qfl_loss = self.qfl(pred_scores, target_quality_scores)
        if self.class_weights is not None:
            qfl_loss *= self.class_weights
        loss[1] = qfl_loss.sum() / target_scores_sum

        # 2. 2D Gaussian Wasserstein Distance Loss
        if fg_mask.sum():
            loss_gwd, loss_unc = self.gwd_bbox_loss(
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
            loss[0] = loss_gwd
            loss[2] = loss_unc

            im_files = batch.get("im_file") or batch.get("im_files") or batch.get("img_files")
            log_per_image_training_variance(pred_distri, fg_mask, im_files, trainer=getattr(self, "trainer", None))

        box_gain = float(getattr(self.hyp, "box", 8.0) if getattr(self.hyp, "box", None) is not None else 8.0)
        cls_gain = float(getattr(self.hyp, "cls", 0.5) if getattr(self.hyp, "cls", None) is not None else 0.5)
        dfl_gain = float(getattr(self.hyp, "dfl", 0.5) if getattr(self.hyp, "dfl", None) is not None else 0.5)

        loss[0] *= box_gain
        loss[1] *= cls_gain
        loss[2] *= dfl_gain

        return (
            (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor),
            loss,
            dict(zip(self.loss_names, loss.detach())),
        )


class v8GaussianGWDQFLE2EDetectLoss(E2ELoss):
    """
    Dual-Head End-to-End Detect Loss using 2D GWD + Uncertainty-Guided Quality Focal Loss (QFL).
    """
    def __init__(self, model, c_scale: float = 0.5, qfl_gamma: float = 1.5):
        loss_fn = (lambda m, tal_topk=10, tal_topk2=None: v8GaussianGWDQFLDetectionLoss(m, tal_topk=tal_topk, tal_topk2=tal_topk2, c_scale=c_scale, qfl_gamma=qfl_gamma))
        super().__init__(model, loss_fn=loss_fn)
        self.one2one = v8GaussianGWDQFLDetectionLoss(model, tal_topk=1, c_scale=c_scale, qfl_gamma=qfl_gamma)


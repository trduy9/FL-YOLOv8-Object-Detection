"""
evaluate_global_model_gaussian.py

This script evaluates all YOLOv8/YOLO26 global model checkpoints in a directory on a validation dataset,
applying the Gaussian Uncertainty Attenuation criterion during postprocessing:

    Score_final = P(Class_c) * (1 - (Sigma_l + Sigma_t + Sigma_r + Sigma_b) / 4)

Usage:
    python evaluate_global_model_gaussian.py --models_dir <dir> --val_yaml <path> --device <cpu/cuda:0> --output_dir <output_dir>
"""

import os
import argparse
import glob
import torch
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionValidator

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate YOLO global models using Gaussian Uncertainty Attenuation Criterion.")
    parser.add_argument('--models_dir', type=str, default='global_model_checkpoints', help='Directory with global model .pt files')
    parser.add_argument('--val_yaml', type=str, default='validation_data.yaml', help='Validation data YAML file')
    parser.add_argument('--device', type=str, default='cpu', help="Device to use: 'cpu' or 'cuda:0'")
    parser.add_argument('--output_dir', type=str, default='evaluation_results_gaussian_attenuated', help='Directory to save evaluation results')
    return parser.parse_args()


class GaussianDetectionValidator(DetectionValidator):
    """
    Custom DetectionValidator that attenuates confidence scores by boundary uncertainty (Sigma)
    prior to NMS filtering.
    """
    def postprocess(self, preds):
        # Check if preds is tuple/list from head
        if isinstance(preds, (list, tuple)):
            preds_tensor = preds[0]
        else:
            preds_tensor = preds

        if isinstance(preds_tensor, torch.Tensor) and preds_tensor.dim() == 3:
            # Check if output channels indicate 8 bbox channels (4 mu + 4 sigma)
            # Shape: (B, C, N) where C = 8 + num_classes
            num_channels = preds_tensor.shape[1]
            if num_channels > 8:
                nc = num_channels - 8
                boxes_mu = preds_tensor[:, :4, :]                              # (B, 4, N)
                boxes_sigma = torch.sigmoid(preds_tensor[:, 4:8, :]).permute(0, 2, 1) # (B, N, 4)
                scores = preds_tensor[:, 8:, :].permute(0, 2, 1)              # (B, N, nc)

                # Uncertainty attenuation: Score_final = P(Class_c) * (1 - avg_sigma)
                avg_sigma = boxes_sigma.mean(dim=-1, keepdim=True)             # (B, N, 1)
                attenuation_factor = (1.0 - avg_sigma).clamp(min=0.0, max=1.0)
                attenuated_scores = (scores * attenuation_factor).permute(0, 2, 1) # (B, nc, N)

                # Reconstruct tensor (B, 4 + nc, N) with attenuated scores for standard NMS
                preds_tensor = torch.cat([boxes_mu, attenuated_scores], dim=1)
                if isinstance(preds, tuple):
                    preds = (preds_tensor,) + preds[1:]
                else:
                    preds = preds_tensor

        return super().postprocess(preds)


def evaluate_model(model_path, validation_data_yaml, device, output_dir):
    print(f"\n--- Evaluating Gaussian Model with Uncertainty Attenuation: {model_path} ---")
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        return
    try:
        model = YOLO(model_path)
        class_names = model.names
        nc = len(class_names)
        print(f"  Number of classes: {nc}")
        print(f"  Class names: {list(class_names.values())}")
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return

    try:
        validator = GaussianDetectionValidator(args=dict(
            data=validation_data_yaml,
            imgsz=512,
            batch=4,
            device=device,
            split='val',
            conf=0.001,
            iou=0.65,
            plots=True,
            project=output_dir,
            name=os.path.basename(model_path).replace('.pt', '')
        ))
        validator(model=model.model)
        results = validator.metrics

        # Print overall metrics
        print(f"  mAP50-95: {results.box.map}")
        print(f"  mAP50:    {results.box.map50}")
        if hasattr(results.box.p, 'mean'):
            print(f"  Precision (mean): {results.box.p.mean()}")
        else:
            print(f"  Precision: {results.box.p}")
        if hasattr(results.box.r, 'mean'):
            print(f"  Recall (mean):    {results.box.r.mean()}")
        else:
            print(f"  Recall:    {results.box.r}")
        if hasattr(results.box.f1, 'mean'):
            print(f"  F1 (mean):        {results.box.f1.mean()}")
        else:
            print(f"  F1:        {results.box.f1}")
    except Exception as e:
        print(f"Error evaluating model {model_path}: {e}")


def main():
    args = parse_args()
    models_dir = args.models_dir
    validation_data_yaml = args.val_yaml
    device = args.device
    output_dir = args.output_dir

    if not os.path.exists(models_dir):
        print(f"Error: Models directory not found: {models_dir}")
        return

    model_paths = sorted(glob.glob(os.path.join(models_dir, "*.pt")))
    if not model_paths:
        print(f"No .pt model files found in {models_dir}")
        return

    print(f"Found {len(model_paths)} models to evaluate with Gaussian Uncertainty Attenuation in {models_dir}:")
    for path in model_paths:
        print(f"  - {os.path.basename(path)}")

    for model_path in model_paths:
        evaluate_model(model_path, validation_data_yaml, device, output_dir)


if __name__ == "__main__":
    main()

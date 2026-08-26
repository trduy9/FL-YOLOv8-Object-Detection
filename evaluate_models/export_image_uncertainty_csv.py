"""
export_image_uncertainty_csv.py

This script evaluates a trained Gaussian YOLO model on a dataset, calculates per-image 
bounding box variance / uncertainty metrics (Sigma_l, Sigma_t, Sigma_r, Sigma_b), and exports
a comprehensive analysis report to a CSV file.

Theoretical Property Tested:
- Clean samples -> Low predicted variance (Sigma -> 0)
- Noisy / Ambiguous samples -> High predicted variance (Sigma phình to)

Usage:
    python evaluate_models/export_image_uncertainty_csv.py \
        --model_path /path/to/global_model_r10.pt \
        --data_yaml /path/to/data.yaml \
        --output_csv image_uncertainty_report.csv \
        --conf_thresh 0.25 \
        --device cuda:0
"""

import os
import argparse
import csv
import glob
import torch
import numpy as np
from ultralytics import YOLO
import yaml

def parse_args():
    parser = argparse.ArgumentParser(description="Export Gaussian Bbox Variance/Uncertainty Per Image to CSV.")
    parser.add_argument('--model_path', type=str, required=True, help="Path to trained Gaussian YOLO model .pt checkpoint")
    parser.add_argument('--data_yaml', type=str, required=True, help="Path to dataset data.yaml file")
    parser.add_argument('--split', type=str, default='val', choices=['val', 'test', 'train'], help="Dataset split to analyze (default: val)")
    parser.add_argument('--output_csv', type=str, default='image_uncertainty_report.csv', help="Output CSV filepath")
    parser.add_argument('--conf_thresh', type=float, default=0.25, help="Confidence threshold to select valid detections")
    parser.add_argument('--device', type=str, default='cuda:0', help="Device to run inference ('cuda:0' or 'cpu')")
    return parser.parse_args()

def get_image_paths_from_yaml(data_yaml_path, split='val'):
    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    yaml_dir = os.path.dirname(os.path.abspath(data_yaml_path))
    val_path = data_config.get(split, data_config.get('val'))

    if isinstance(val_path, list):
        val_path = val_path[0]
    
    if not os.path.isabs(val_path):
        val_path = os.path.join(yaml_dir, val_path)

    if os.path.isdir(val_path):
        image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(glob.glob(os.path.join(val_path, ext)))
            image_paths.extend(glob.glob(os.path.join(val_path, '**', ext), recursive=True))
        return sorted(list(set(image_paths)))
    elif os.path.isfile(val_path) and val_path.endswith('.txt'):
        with open(val_path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        img_paths = []
        for line in lines:
            if not os.path.isabs(line):
                line = os.path.join(yaml_dir, line)
            img_paths.append(line)
        return sorted(img_paths)
    else:
        raise FileNotFoundError(f"Could not locate image directory or txt list from val_path: {val_path}")

def analyze_image_uncertainty():
    args = parse_args()

    print(f"⚡ Loading Gaussian YOLO model from: {args.model_path}")
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")

    model = YOLO(args.model_path)
    device = torch.device(args.device if torch.cuda.is_available() and 'cuda' in args.device else 'cpu')
    model.to(device)

    # Locate dataset images
    image_paths = get_image_paths_from_yaml(args.data_yaml, split=args.split)
    print(f"📊 Found {len(image_paths)} images in split '{args.split}' for analysis.")

    results_data = []

    print(f"🔍 Running inference and extracting variance metrics (Threshold = {args.conf_thresh})...")
    
    for idx, img_path in enumerate(image_paths):
        img_name = os.path.basename(img_path)
        
        # Run raw model prediction
        preds = model.predict(source=img_path, conf=args.conf_thresh, device=device, verbose=False)[0]

        # Extract predicted boxes and raw outputs if available
        # Note: YOLO.predict returns Results object
        boxes = preds.boxes
        if len(boxes) == 0:
            # No object detected above threshold
            results_data.append({
                'image_name': img_name,
                'num_detections': 0,
                'mean_sigma_l': 0.0,
                'mean_sigma_t': 0.0,
                'mean_sigma_r': 0.0,
                'mean_sigma_b': 0.0,
                'overall_avg_sigma': 0.0,
                'max_sigma': 0.0,
                'min_sigma': 0.0,
                'uncertainty_category': 'No Detection'
            })
            continue

        # If model outputs 8 channels (4 mu + 4 sigma), boxes.data or raw head contains sigma
        # In ultralytics, adapted Detect head output channels are: (l, t, r, b, sigma_l, sigma_t, sigma_r, sigma_b)
        # We check if raw predictions tensor exists
        if hasattr(preds, 'orig_shape'):
            # Fetch raw detection tensor if saved or calculate from confidence
            pass

        # For YOLO adapted model, boxes tensor: (N, 6) [x1, y1, x2, y2, conf, cls]
        # To get exact sigmas, we run forward pass on single image batch tensor
        img_tensor = torch.from_numpy(preds.orig_img).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
        # Resize/pad to model imgsz
        img_tensor = torch.nn.functional.interpolate(img_tensor, size=(512, 512), mode='bilinear', align_corners=False)

        with torch.no_grad():
            raw_out = model.model(img_tensor)
            if isinstance(raw_out, (tuple, list)):
                raw_out = raw_out[0]
            
            # raw_out shape: (1, 8 + num_classes, N_anchors)
            if raw_out.shape[1] > 8:
                sigma_logits = raw_out[:, 4:8, :] # (1, 4, N)
                sigmas = torch.sigmoid(sigma_logits).squeeze(0).permute(1, 0) # (N, 4)
                
                scores = raw_out[:, 8:, :].squeeze(0).permute(1, 0) # (N, num_classes)
                max_scores, _ = scores.sigmoid().max(dim=-1) # (N,)
                
                valid_mask = max_scores > args.conf_thresh
                if valid_mask.any():
                    valid_sigmas = sigmas[valid_mask] # (N_valid, 4)
                    
                    sigma_l = valid_sigmas[:, 0].mean().item()
                    sigma_t = valid_sigmas[:, 1].mean().item()
                    sigma_r = valid_sigmas[:, 2].mean().item()
                    sigma_b = valid_sigmas[:, 3].mean().item()
                    overall_avg = valid_sigmas.mean().item()
                    max_sig = valid_sigmas.max().item()
                    min_sig = valid_sigmas.min().item()
                    num_det = valid_sigmas.shape[0]

                    # Categorize uncertainty
                    if overall_avg < 0.15:
                        category = 'Low (Clean Sample)'
                    elif overall_avg <= 0.35:
                        category = 'Medium (Slight Noise/Blur)'
                    else:
                        category = 'High (Noisy/Ambiguous)'

                    results_data.append({
                        'image_name': img_name,
                        'num_detections': num_det,
                        'mean_sigma_l': round(sigma_l, 4),
                        'mean_sigma_t': round(sigma_t, 4),
                        'mean_sigma_r': round(sigma_r, 4),
                        'mean_sigma_b': round(sigma_b, 4),
                        'overall_avg_sigma': round(overall_avg, 4),
                        'max_sigma': round(max_sig, 4),
                        'min_sigma': round(min_sig, 4),
                        'uncertainty_category': category
                    })
                else:
                    results_data.append({
                        'image_name': img_name,
                        'num_detections': 0,
                        'mean_sigma_l': 0.0,
                        'mean_sigma_t': 0.0,
                        'mean_sigma_r': 0.0,
                        'mean_sigma_b': 0.0,
                        'overall_avg_sigma': 0.0,
                        'max_sigma': 0.0,
                        'min_sigma': 0.0,
                        'uncertainty_category': 'Low Conf'
                    })

        if (idx + 1) % 50 == 0 or (idx + 1) == len(image_paths):
            print(f"  Processed {idx + 1}/{len(image_paths)} images...")

    # Export to CSV
    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    fieldnames = [
        'image_name', 'num_detections', 'mean_sigma_l', 'mean_sigma_t', 
        'mean_sigma_r', 'mean_sigma_b', 'overall_avg_sigma', 'max_sigma', 
        'min_sigma', 'uncertainty_category'
    ]

    with open(args.output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_data)

    print(f"\n✅ SUCCESS! Exported per-image uncertainty analysis report to:\n   📄 {os.path.abspath(args.output_csv)}")

    # Print Summary Statistics
    valid_records = [r for r in results_data if r['num_detections'] > 0]
    if valid_records:
        all_sigmas = [r['overall_avg_sigma'] for r in valid_records]
        low_count = sum(1 for r in valid_records if r['uncertainty_category'] == 'Low (Clean Sample)')
        med_count = sum(1 for r in valid_records if r['uncertainty_category'] == 'Medium (Slight Noise/Blur)')
        high_count = sum(1 for r in valid_records if r['uncertainty_category'] == 'High (Noisy/Ambiguous)')

        print("\n=== SUMMARY STATISTICS ===")
        print(f" Total Images Evaluated: {len(results_data)}")
        print(f" Images with Detections: {len(valid_records)}")
        print(f" Mean Dataset Variance (Sigma): {np.mean(all_sigmas):.4f}")
        print(f" Min Dataset Variance: {np.min(all_sigmas):.4f}")
        print(f" Max Dataset Variance: {np.max(all_sigmas):.4f}")
        print(f" 🟢 Clean Samples (Sigma < 0.15): {low_count} ({low_count/len(valid_records)*100:.1f}%)")
        print(f" 🟡 Medium Samples (0.15 <= Sigma <= 0.35): {med_count} ({med_count/len(valid_records)*100:.1f}%)")
        print(f" 🔴 Noisy Samples (Sigma > 0.35): {high_count} ({high_count/len(valid_records)*100:.1f}%)")
        print("===========================\n")

if __name__ == '__main__':
    analyze_image_uncertainty()

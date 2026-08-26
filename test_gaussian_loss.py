import os
import sys
import torch
import torch.nn.functional as F

# Ensure core is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.gaussian_loss import GaussianBboxLoss, v8GaussianDetectionLoss
from core.gaussian_yolo import adapt_yolo26_for_gaussian, apply_gaussian_inference_criterion, get_detect_head
from ultralytics import YOLO

def test_gaussian_bbox_loss():
    print("=== Test 1: GaussianBboxLoss Forward & Backward Pass ===")
    loss_fn = GaussianBboxLoss()
    
    batch_size = 2
    num_anchors = 100
    
    # Dummy predictions (B, N, 8)
    pred_dist = torch.randn(batch_size, num_anchors, 8, requires_grad=True)
    pred_bboxes = torch.rand(batch_size, num_anchors, 4) * 100
    anchor_points = torch.rand(num_anchors, 2) * 10
    target_bboxes = torch.rand(batch_size, num_anchors, 4) * 100
    target_scores = torch.rand(batch_size, num_anchors, 80)
    target_scores_sum = torch.tensor(10.0)
    
    fg_mask = torch.zeros(batch_size, num_anchors, dtype=torch.bool)
    fg_mask[0, :10] = True
    fg_mask[1, 5:15] = True
    
    imgsz = torch.tensor([640.0, 640.0])
    stride = torch.ones(num_anchors, 1) * 8.0
    
    loss_g, loss_l1 = loss_fn(
        pred_dist=pred_dist,
        pred_bboxes=pred_bboxes,
        anchor_points=anchor_points,
        target_bboxes=target_bboxes,
        target_scores=target_scores,
        target_scores_sum=target_scores_sum,
        fg_mask=fg_mask,
        imgsz=imgsz,
        stride=stride
    )
    
    print(f"Gaussian NLL Loss: {loss_g.item():.6f}, L1 Loss: {loss_l1.item():.6f}")
    assert not torch.isnan(loss_g), "Loss is NaN!"
    assert not torch.isinf(loss_g), "Loss is Inf!"
    
    (loss_g + loss_l1).backward()
    assert pred_dist.grad is not None, "Gradient was not computed!"
    print("✅ Test 1 PASSED: GaussianBboxLoss & L1 loss calculation & gradients working.")

def test_model_adaptation():
    print("\n=== Test 2: YOLO26 Model Adaptation ===")
    weights_path = "/media/data3/home/truongduy/FL-YOLOv8-Object-Detection/yolo26n.pt"
    if not os.path.exists(weights_path):
        weights_path = "yolo11n.pt"
        
    model = YOLO(weights_path)
    adapt_yolo26_for_gaussian(model)
    
    # Verify cv2 outputs 8 channels
    detect_head = get_detect_head(model)
    assert detect_head is not None, "Could not find Detect head."
    for i, seq in enumerate(detect_head.cv2):
        out_ch = seq[2].out_channels
        print(f"Detect head cv2[{i}][2] output channels: {out_ch}")
        assert out_ch == 8, f"Expected 8 output channels, got {out_ch}"
        
    print("✅ Test 2 PASSED: YOLO26 Detect head adapted to 8 channels.")

def test_inference_criterion():
    print("\n=== Test 3: Gaussian Inference Detection Criterion ===")
    pred_scores = torch.tensor([[0.9, 0.8], [0.5, 0.4]]) # (2, 2)
    # High uncertainty (Sigma ~ 0.8) vs Low uncertainty (Sigma ~ 0.1)
    pred_sigma = torch.tensor([
        [0.8, 0.8, 0.8, 0.8], # avg = 0.8 -> factor = 0.2
        [0.1, 0.1, 0.1, 0.1]  # avg = 0.1 -> factor = 0.9
    ])
    
    final_scores = apply_gaussian_inference_criterion(pred_scores, pred_sigma)
    print(f"Raw scores:\n{pred_scores}")
    print(f"Final scores after uncertainty attenuation:\n{final_scores}")
    
    # High uncertainty score should drop from 0.9 to 0.9 * 0.2 = 0.18
    assert torch.isclose(final_scores[0, 0], torch.tensor(0.18)), "High uncertainty drop mismatch!"
    # Low uncertainty score should drop from 0.5 to 0.5 * 0.9 = 0.45
    assert torch.isclose(final_scores[1, 0], torch.tensor(0.45)), "Low uncertainty drop mismatch!"
    print("✅ Test 3 PASSED: Inference criterion attenuation verified.")

if __name__ == "__main__":
    test_gaussian_bbox_loss()
    test_model_adaptation()
    test_inference_criterion()
    print("\n🎉 ALL UNIT TESTS PASSED SUCCESSFULLY!")

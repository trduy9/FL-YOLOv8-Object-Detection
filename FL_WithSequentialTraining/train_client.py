import os
import sys

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ultralytics import YOLO
from core.gaussian_yolo import adapt_yolo26_for_gaussian, GaussianCIoUE2EDetectionTrainer

def train_client(
    data_path,
    client_run_name,
    epochs=10,
    weights_path="/media/data3/home/truongduy/FL-YOLOv8-Object-Detection/yolo26m.pt",
    device='cpu',
    workers=0,
    output_dir='runs/detect',
    use_gaussian_loss=False
):
    """
    Train YOLOv8 / YOLO26 model on a client's local data with option to use custom Gaussian Bbox Loss.

    Args:
        data_path (str): Path to client dataset YAML file.
        client_run_name (str): Name to save the trained model weights.
        epochs (int): Number of training epochs.
        weights_path (str): Path to starting weights file.
        device (str): Device to use for training (default: cpu).
        workers (int): Number of dataloader workers (default: 0).
        output_dir (str): Directory to save the YOLO run (default: runs/detect).
        use_gaussian_loss (bool): Whether to use custom NLL Gaussian bbox loss.
    """
    model = YOLO(weights_path)  # Load specified weights
    output_dir = os.path.abspath(output_dir)

    train_kwargs = {
        "data": data_path,
        "epochs": epochs,
        "imgsz": 512,
        "batch": 4,
        "name": client_run_name,
        "project": output_dir,
        "save": True,
        "device": device,
        "workers": workers,
        "val": False,
        "exist_ok": True,
        "box": 7.5,      # Trọng số 7.5 cho Gaussian NLL Bbox Loss
        "dfl": 1.5,      # Trọng số 1.5 cho CIoU Loss

    }

    if use_gaussian_loss:
        print("⚡ Custom Gaussian NLL (7.5) + CIoU (1.5) E2E Loss enabled for YOLO26 training!")
        adapt_yolo26_for_gaussian(model)
        train_kwargs["trainer"] = GaussianCIoUE2EDetectionTrainer

    # Train model
    model.train(**train_kwargs)

    print(f"Training completed for client. Model saved at {os.path.join(output_dir, client_run_name, 'weights', 'best.pt')}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help="Path to client's data YAML file")
    parser.add_argument('--save', type=str, required=True, help="Name to save the trained model")
    parser.add_argument('--epochs', type=int, default=10, help="Number of epochs to train")
    parser.add_argument('--weights', type=str, default="/media/data3/home/truongduy/FL-YOLOv8-Object-Detection/yolo26m.pt", help="Path to starting weights")
    parser.add_argument('--device', type=str, default='cpu', help="Device to train on ('cpu' or 'cuda')")
    parser.add_argument('--workers', type=int, default=0, help="Number of dataloader workers")
    parser.add_argument('--output_dir', type=str, default='runs/detect', help="Directory to save the trained model")
    parser.add_argument('--use_gaussian_loss', action='store_true', help="Use custom NLL Gaussian Bbox Loss")

    args = parser.parse_args()

    train_client(
        data_path=args.data,
        client_run_name=args.save,
        epochs=args.epochs,
        weights_path=args.weights,
        device=args.device,
        workers=args.workers,
        output_dir=args.output_dir,
        use_gaussian_loss=args.use_gaussian_loss
    )

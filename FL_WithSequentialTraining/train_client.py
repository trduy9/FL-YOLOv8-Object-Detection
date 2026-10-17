import os
import sys

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ultralytics import YOLO
from core.gaussian_yolo import adapt_yolo26_for_gaussian, GaussianDetectionTrainer
from core.gaussian_loss import flush_variance_log, _parse_client_round

def train_client(
    data_path,
    client_run_name,
    epochs=10,
    weights_path="/media/data3/home/truongduy/FL-YOLOv8-Object-Detection/yolo26n.pt",
    device='cpu',
    workers=0,
    output_dir='runs/detect',
    use_gaussian_loss=False,
    loss_type="ciou",
    fedprox_mu=0.0,
    global_weights_path=None
):
    """
    Train YOLOv8 / YOLO26 model on a client's local data with option to use custom Gaussian Bbox Loss
    and FedProx proximal term penalty.

    Args:
        data_path (str): Path to client dataset YAML file.
        client_run_name (str): Name to save the trained model weights.
        epochs (int): Number of training epochs.
        weights_path (str): Path to starting weights file.
        device (str): Device to use for training (default: cpu).
        workers (int): Number of dataloader workers (default: 0).
        output_dir (str): Directory to save the YOLO run (default: runs/detect).
        use_gaussian_loss (bool): Whether to use custom NLL Gaussian bbox loss.
        loss_type (str): Type of Gaussian loss to use (ciou, gwd, gwd_qfl).
        fedprox_mu (float): FedProx proximal term coefficient (mu >= 0).
        global_weights_path (str): Path to the global model checkpoint for FedProx proximal reference.
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
        "box": 8.0,
        "dfl": 1.0,
    }

    base_trainer_cls = None
    if use_gaussian_loss:
        adapt_yolo26_for_gaussian(model)
        if loss_type == "gwd_qfl":
            from core.gaussian_yolo import GaussianGWDQFLDetectionTrainer
            print("⚡ Custom 2D GWD + Quality Focal Loss (QFL) enabled for YOLO26 training!")
            base_trainer_cls = GaussianGWDQFLDetectionTrainer
        elif loss_type == "gwd":
            from core.gaussian_yolo import GaussianGWDDetectionTrainer
            print("⚡ Custom 2D Gaussian Wasserstein Distance (GWD) Loss enabled for YOLO26 training!")
            base_trainer_cls = GaussianGWDDetectionTrainer
        else:
            print("⚡ Custom Gaussian Loss (CIoU + Kendall & Gal NLL) enabled for YOLO26 training!")
            base_trainer_cls = GaussianDetectionTrainer

    # Kích hoạt FedProx Proximal Term nếu được bật (fedprox_mu > 0 và có global_weights_path)
    if fedprox_mu > 0 and global_weights_path:
        from core.fedprox import load_global_model_weights, make_fedprox_trainer_class
        from ultralytics.models.yolo.detect import DetectionTrainer
        if base_trainer_cls is None:
            base_trainer_cls = DetectionTrainer
        global_state = load_global_model_weights(global_weights_path)
        print(f"⚡ FedProx Proximal Term activated: mu={fedprox_mu}, global_weights={global_weights_path}")
        train_kwargs["trainer"] = make_fedprox_trainer_class(base_trainer_cls, global_state, fedprox_mu)
    elif base_trainer_cls is not None:
        train_kwargs["trainer"] = base_trainer_cls

    # Train model
    model.train(**train_kwargs)

    # Flush accumulated variance/mu stats to CSV after training
    if use_gaussian_loss:
        client_name, round_num = _parse_client_round(client_run_name)
        client_run_dir = os.path.join(output_dir, client_run_name)
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        flush_variance_log(
            client_name=client_name,
            round_num=round_num,
            last_epoch=epochs - 1,
            save_dir=client_run_dir,
            project_root=project_root
        )

    print(f"Training completed for client. Model saved at {os.path.join(output_dir, client_run_name, 'weights', 'best.pt')}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help="Path to client's data YAML file")
    parser.add_argument('--save', type=str, required=True, help="Name to save the trained model")
    parser.add_argument('--epochs', type=int, default=10, help="Number of epochs to train")
    parser.add_argument('--weights', type=str, default="/media/data3/home/truongduy/FL-YOLOv8-Object-Detection/yolo26n.pt", help="Path to starting weights")
    parser.add_argument('--device', type=str, default='cpu', help="Device to train on ('cpu' or 'cuda')")
    parser.add_argument('--workers', type=int, default=0, help="Number of dataloader workers")
    parser.add_argument('--output_dir', type=str, default='runs/detect', help="Directory to save the trained model")
    parser.add_argument('--use_gaussian_loss', action='store_true', help="Use custom NLL Gaussian Bbox Loss")
    parser.add_argument('--loss_type', type=str, default='ciou', choices=['ciou', 'gwd', 'gwd_qfl'], help="Gaussian loss variant: 'ciou', 'gwd', or 'gwd_qfl'")
    parser.add_argument('--fedprox_mu', type=float, default=0.0, help="FedProx proximal term coefficient (default: 0.0, disabled)")
    parser.add_argument('--global_weights', type=str, default=None, help="Path to global model checkpoint for FedProx proximal reference")

    args = parser.parse_args()

    train_client(
        data_path=args.data,
        client_run_name=args.save,
        epochs=args.epochs,
        weights_path=args.weights,
        device=args.device,
        workers=args.workers,
        output_dir=args.output_dir,
        use_gaussian_loss=args.use_gaussian_loss,
        loss_type=args.loss_type,
        fedprox_mu=args.fedprox_mu,
        global_weights_path=args.global_weights
    )

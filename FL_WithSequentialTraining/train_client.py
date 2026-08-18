import os
from ultralytics import YOLO

def train_client(data_path, client_run_name, epochs=10, weights_path="/media/data3/home/truongduy/FL-YOLOv8-Object-Detection/yolo26m.pt", device='cpu', workers=0, output_dir='runs/detect'):
    """
    Train YOLOv8 model on a client's local data.

    Args:
        data_path (str): Path to client dataset YAML file.
        client_run_name (str): Name to save the trained model weights.
        epochs (int): Number of training epochs.
        weights_path (str): Path to starting weights file (default: yolov8n.pt).
        device (str): Device to use for training (default: cpu).
        workers (int): Number of dataloader workers (default: 1).
        output_dir (str): Directory to save the YOLO run (default: runs/detect)
    """
    model = YOLO(weights_path)  # Load specified weights!
    output_dir = os.path.abspath(output_dir)

    # Train on client's data
    model.train(
        data=data_path,
        epochs=epochs,
        imgsz=512,
        batch=4,
        name=client_run_name,
        project=output_dir,
        save=True,
        device=device,
        workers=workers,
        val=False,
        exist_ok=True
    )
#    model.val(  
#        data=data_path,  
#        imgsz=640,  
#        device=device,  
#        split='val'  
#    )

    print(f"Training completed for client. Model saved at {os.path.join(output_dir, client_run_name, 'weights', 'best.pt')}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help="Path to client's data YAML file")
    parser.add_argument('--save', type=str, required=True, help="Name to save the trained model")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs")
    parser.add_argument('--weights', type=str, default="/media/data3/home/truongduy/FL-YOLOv8-Object-Detection/yolo26m.pt", help="Starting weights path (default: yolov8n.pt)")
    parser.add_argument('--device', type=str, default='cpu', help="Device to use for training (default: cpu)")
    parser.add_argument('--workers', type=int, default=1, help="Number of dataloader workers (default: 1)")
    parser.add_argument('--output_dir', type=str, default='runs/detect', help="Directory to save the YOLO run (default: runs/detect)")
    args = parser.parse_args()

    train_client(args.data, args.save, args.epochs, args.weights, args.device, args.workers, args.output_dir)

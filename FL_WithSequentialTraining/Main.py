"""
Main.py

This script simulates federated learning with sequential client training using YOLOv8 for object detection.
Configuration is loaded from config.yaml in this folder.

Usage:
    python Main.py
    (Edit FL_WithSequentialTraining/config.yaml as needed)
"""

import os
import sys
import subprocess
import torch
import yaml
import logging
import time
import csv  
import random  
import glob 
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.insert(0, project_root)          
  
from ultralytics import YOLO  
  
# Import modular components  
from core.aggregation import federated_average, fedprox_aggregate  
from core.utils import adapt_and_save_initial_model  
from core.oort.oort_wrapper import OortClientSampler

  
 



# === LOGGING SETUP ===
# Explicitly set up logging to handle unicode characters in the log file
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create a formatter
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')

# Create a file handler with utf-8 encoding
file_handler = logging.FileHandler('federated_learning.log', encoding='utf-8')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Create a stream handler (for console output)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)



  

  
def count_images(data_yaml):  
    with open(data_yaml) as f:  
        d = yaml.safe_load(f)  
    train_path = d.get('train', '')  
    exts = ('*.jpg', '*.jpeg', '*.png')  
    n = sum(len(glob.glob(os.path.join(train_path, e))) for e in exts)  
    return n or 1  
  
  
def read_loss_from_csv(client_run_dir):  
    results_csv = os.path.join(client_run_dir, 'results.csv')  
    if not os.path.exists(results_csv):  
        return 2.0  
    with open(results_csv) as f:  
        rows = list(csv.DictReader(f))  
    if not rows:  
        return 2.0  
    last = rows[-1]  
    
    try:
        box_loss = float(last.get('train/box_loss', 0.0) or 0.0)
        cls_loss = float(last.get('train/cls_loss', 0.0) or 0.0)
        dfl_loss = float(last.get('train/dfl_loss', 0.0) or 0.0)
        
        # When Gaussian loss is used, dfl_loss represents Gaussian NLL which can be negative (ln(sigma) < 0).
        # Oort statistical utility requires a non-negative loss (loss >= 0) to compute sqrt(loss).
        if dfl_loss < 0:
            total_loss = box_loss + cls_loss
        else:
            total_loss = box_loss + cls_loss + dfl_loss
            
        return max(total_loss, 1e-4)
    except Exception:
        return 2.0  
  

  
  
def select_client_ids(round_num):  
    all_ids = list(id_to_client.keys())  
  
    if SELECTION_METHOD == 'all':  
        return all_ids  
  
    # warmup 
    if round_num < OORT_START_ROUND:  
        logger.info(f"[Warmup] Round {round_num + 1}: training ALL {len(all_ids)} clients")  
        return all_ids  
  
    if SELECTION_METHOD == 'random':  
        selected = _rng.sample(all_ids, min(NUM_CLIENTS_PER_ROUND, len(all_ids)))  
        logger.info(f"[Random] Round {round_num + 1} selected: {selected}")  
        return selected  
  
    if SELECTION_METHOD == 'oort':  
        selected = oort.select_clients(  
            num_clients=NUM_CLIENTS_PER_ROUND,  
            feasible_clients=all_ids,  
            round_num=round_num  
        )  
        logger.info(f"[Oort] Round {round_num + 1} selected: {selected}")  
        return selected  
  
    return all_ids

# === CONFIG LOADING ===
def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

config = load_config(os.path.join(os.path.dirname(__file__), 'config.yaml'))

NUM_ROUNDS = config.get('num_rounds', 3)
EPOCHS_PER_ROUND = config.get('epochs_per_round', 10)
NUM_CLASSES = config.get('num_classes')
FL_PROJECT_OUTPUTS_DIR = config.get('fl_project_outputs_dir', 'FL_YOLOv8_Project_Outputs')
CLIENTS = config.get('clients', [])
DEVICE = config.get('device', 'cpu') # Get device from config
USE_GAUSSIAN_LOSS = config.get('use_gaussian_loss', False)
GAUSSIAN_LOSS_TYPE = config.get('gaussian_loss_type', 'ciou')

# --- Federated Learning Algorithm (fedavg | fedprox) ---
FL_ALGORITHM = str(config.get('fl_algorithm', config.get('aggregation_method', 'fedavg'))).lower().strip()
FEDPROX_MU = float(config.get('fedprox_mu', 0.01))

# --- Cau hinh chon client ---  
SELECTION_METHOD = config.get('selection_method', 'all')   # oort | random | all  
NUM_CLIENTS_PER_ROUND = config.get('num_clients_per_round', len(CLIENTS))  
OORT_START_ROUND = config.get('oort_start_round', 5)  
SAMPLE_SEED = 233

_rng = random.Random(SAMPLE_SEED)  
id_to_client = {i: c for i, c in enumerate(CLIENTS)}   

oort = None  
if SELECTION_METHOD == 'oort':  
    oort = OortClientSampler({'sample_seed': SAMPLE_SEED})  
    for cid, c in id_to_client.items():  
        oort.register_client(cid, data_size=count_images(c['data_yaml']), duration=2000)  
  

# Output directory for global model checkpoints
GLOBAL_MODEL_SAVE_DIR = os.path.join(FL_PROJECT_OUTPUTS_DIR, 'global_model_checkpoints')
os.makedirs(GLOBAL_MODEL_SAVE_DIR, exist_ok=True)  # Ensure directory exists
INITIAL_GLOBAL_MODEL_PATH = os.path.join(GLOBAL_MODEL_SAVE_DIR, 'global_model_r0.pt')

# Determine number of workers for client data loaders
cpu_count = os.cpu_count()
if cpu_count is None or cpu_count < 2:
    NUM_WORKERS_FOR_CLIENT_DATALOADERS = 1
else:
    NUM_WORKERS_FOR_CLIENT_DATALOADERS = cpu_count // 2

#def train_clients(round_num):
#    logger.info(f"=== Training Round {round_num + 1} ===")
#    weights_paths = []
#    current_global_model_path = os.path.join(GLOBAL_MODEL_SAVE_DIR, f'global_model_r{round_num}.pt')
#
#    # --- Path to the training script, robustly defined ---
#    script_dir = os.path.dirname(__file__)
#    train_client_script_path = os.path.join(script_dir, 'train_client.py')
#    client_models_dir = os.path.join(FL_PROJECT_OUTPUTS_DIR, 'client_models')
#    os.makedirs(client_models_dir, exist_ok=True)
#    
#    if round_num < OORT_START_ROUND:  
#        selected_ids = list(id_to_client.keys())  
#    else:  
#        selected_ids = oort.select_clients(  
#            num_clients=NUM_CLIENTS_PER_ROUND,  
#            feasible_clients=list(id_to_client.keys()),  
#            round_num=round_num  
#        )  
#    logger.info(f"[Oort] Round {round_num + 1} selected clients: {selected_ids}")  
#
#    for client in CLIENTS:
#        client = id_to_client[cid]  
#        save_name = f"{client['model_name']}_round{round_num + 1}"
#        if round_num == 0:
#            starting_weights = INITIAL_GLOBAL_MODEL_PATH
#        else:
#            starting_weights = current_global_model_path
#        logger.info(f"{client['model_name']} starting with weights: {starting_weights}")
#        try:
#            subprocess.run([
#                'python', train_client_script_path,
#                '--data', client['data_yaml'],
#                '--save', save_name,
#                '--epochs', str(EPOCHS_PER_ROUND),
#                '--weights', starting_weights,
#                '--device', DEVICE,
#                '--workers', str(NUM_WORKERS_FOR_CLIENT_DATALOADERS),
#                '--output_dir', client_models_dir
#            ], check=True)
#        except subprocess.CalledProcessError as e:
#            logger.error(f"Training failed for client {client['model_name']}: {e}")
#            continue
#        client_run_dir = os.path.join(client_models_dir, save_name)
#        best_pt_path = os.path.join(client_run_dir, 'weights', 'best.pt')
#        if not os.path.exists(best_pt_path):
#            logger.error(f"Client {client['model_name']} trained model not found at {best_pt_path}")
#            continue
#        weights_paths.append(best_pt_path)
#    return weights_paths


def train_clients(round_num):  
    logger.info(f"=== Training Round {round_num + 1} ({FL_ALGORITHM.upper()}) ===")  
    weights_paths = []  
    participating_client_ids = []
    current_global_model_path = os.path.join(GLOBAL_MODEL_SAVE_DIR, f'global_model_r{round_num}.pt')  
  
    # --- Path to the training script, robustly defined ---  
    script_dir = os.path.dirname(__file__)  
    train_client_script_path = os.path.join(script_dir, 'train_client.py')  
    client_models_dir = os.path.join(FL_PROJECT_OUTPUTS_DIR, 'client_models')  
    os.makedirs(client_models_dir, exist_ok=True)  
  
    # --- Chọn client theo phương pháp cấu hình (oort | random | all) ---  
    selected_ids = select_client_ids(round_num)  
  
    for cid in selected_ids:  
        client = id_to_client[cid]  
        save_name = f"{client['model_name']}_round{round_num + 1}"  
        if round_num == 0:  
            starting_weights = INITIAL_GLOBAL_MODEL_PATH  
        else:  
            starting_weights = current_global_model_path  
        logger.info(f"{client['model_name']} starting with weights: {starting_weights}")  
  
        t0 = time.time()  
        try:  
            cmd = [  
                sys.executable, train_client_script_path,  
                '--data', client['data_yaml'],  
                '--save', save_name,  
                '--epochs', str(EPOCHS_PER_ROUND),  
                '--weights', starting_weights,  
                '--device', DEVICE,  
                '--workers', str(NUM_WORKERS_FOR_CLIENT_DATALOADERS),  
                '--output_dir', client_models_dir  
            ]
            if USE_GAUSSIAN_LOSS:
                cmd.append('--use_gaussian_loss')
                if GAUSSIAN_LOSS_TYPE:
                    cmd.extend(['--loss_type', str(GAUSSIAN_LOSS_TYPE)])
            if FL_ALGORITHM == 'fedprox' and FEDPROX_MU > 0:
                cmd.extend(['--fedprox_mu', str(FEDPROX_MU), '--global_weights', starting_weights])
            subprocess.run(cmd, check=True)  
        except subprocess.CalledProcessError as e:  
            logger.error(f"Training failed for client {client['model_name']}: {e}")  
            continue  
        duration = time.time() - t0  
  
        client_run_dir = os.path.join(client_models_dir, save_name)  
        best_pt_path = os.path.join(client_run_dir, 'weights', 'best.pt')  
        if not os.path.exists(best_pt_path):  
            logger.error(f"Client {client['model_name']} trained model not found at {best_pt_path}")  
            continue  
  
        # --- Cập nhật Oort (chỉ khi dùng phương pháp oort) ---  
        if SELECTION_METHOD == 'oort':  
            loss = read_loss_from_csv(client_run_dir)  
            oort.update_client(cid, loss=loss, duration=duration, round_num=round_num)  
            logger.info(f"[Oort] Updated client {cid}: loss={loss:.4f}, duration={duration:.1f}s")  
  
        weights_paths.append(best_pt_path)  
        participating_client_ids.append(cid)
  
    return weights_paths, participating_client_ids

def aggregate(weights_paths, participating_client_ids, round_num):
    logger.info(f"=== Aggregating Models for Round {round_num + 1} ({FL_ALGORITHM.upper()}) ===")
    
    # 1. Load the state_dicts from client models
    client_state_dicts = []
    for p in weights_paths:
        ckpt = torch.load(p, map_location='cpu')
        if 'model' in ckpt and hasattr(ckpt['model'], 'state_dict'):
            client_state_dicts.append(ckpt['model'].state_dict())
        elif 'state_dict' in ckpt:
             client_state_dicts.append(ckpt['state_dict'])
        else:
            # Fallback for simpler .pt files that are just the state_dict
            client_state_dicts.append(ckpt)

    if not client_state_dicts:
        logger.error("No state_dicts loaded, cannot aggregate.")
        return

    # 2. Average the state_dicts using the chosen FL algorithm
    if FL_ALGORITHM == 'fedprox':
        sample_counts = [count_images(id_to_client[cid]['data_yaml']) for cid in participating_client_ids]
        client_names = [id_to_client[cid]['model_name'] for cid in participating_client_ids]
        logger.info(f"⚡ [FedProx Aggregation] Weighting by samples: {dict(zip(client_names, sample_counts))}")
        avg_state_dict = fedprox_aggregate(client_state_dicts, sample_counts=sample_counts)
    else:
        logger.info(f"⚡ [FedAvg Aggregation] Uniform averaging across {len(client_state_dicts)} clients")
        avg_state_dict = federated_average(client_state_dicts)

    # 3. Create a new global model and load the averaged weights
    try:
        base_model_for_aggregation = YOLO(weights_paths[0]) # Load a client model to get the structure
        if hasattr(base_model_for_aggregation, 'model') and base_model_for_aggregation.model is not None:
            base_model_for_aggregation.model.load_state_dict(avg_state_dict)
            next_global_model_path = os.path.join(GLOBAL_MODEL_SAVE_DIR, f'global_model_r{round_num + 1}.pt')
            base_model_for_aggregation.save(next_global_model_path)
            logger.info(f"✅ Aggregated global model for Round {round_num + 1} saved to: {next_global_model_path}")
        else:
            logger.error("Could not load state_dict, base_model_for_aggregation.model is not a valid module.")
    except Exception as e:
        logger.error(f"Failed to save aggregated global model: {e}")

def main():
    logger.info(f"🚀 Federated Learning Pipeline Initialized")
    logger.info(f"   Algorithm: {FL_ALGORITHM.upper()}")
    if FL_ALGORITHM == 'fedprox':
        logger.info(f"   FedProx mu: {FEDPROX_MU} (Proximal regularization term enabled)")
    else:
        logger.info(f"   FedAvg: Standard federated averaging (mu = 0.0, uniform aggregation)")

    # --- Initial Model Adaptation ---
    adapt_and_save_initial_model(
        num_classes=NUM_CLASSES,
        data_yaml_path=CLIENTS[0]['data_yaml'],
        save_path=INITIAL_GLOBAL_MODEL_PATH,
        device=DEVICE
    )

    # --- Federated Learning Rounds ---
    for round_num in range(NUM_ROUNDS):
        logger.info(f"===== Federated Learning Round {round_num + 1}/{NUM_ROUNDS} [{FL_ALGORITHM.upper()}] =====")
        weights_paths, participating_ids = train_clients(round_num)
        if weights_paths:
            aggregate(weights_paths, participating_ids, round_num)
        else:
            logger.warning(f"No client weights to aggregate for round {round_num + 1}. Skipping aggregation.")
        logger.info(f"✅ Completed Round {round_num + 1}\n")

if __name__ == "__main__":
    main()

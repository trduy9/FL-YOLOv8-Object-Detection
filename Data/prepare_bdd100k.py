#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BDD100K Dataset Preparation and Partitioning for Federated Learning (YOLOv8 / YOLO26).
Partitioning logic is identical to fedpylot/datasets/prepare_kitti.py (supports IID and Dirichlet Non-IID).
"""

import argparse
import os
import random
import shutil
import tarfile
import numpy as np
import pandas as pd
import yaml
from collections import defaultdict
from tqdm import tqdm


# BDD100K 10 classes
DEFAULT_CLASS_MAP = {
    'person': 0,
    'rider': 1,
    'car': 2,
    'bus': 3,
    'truck': 4,
    'bike': 5,
    'motor': 6,
    'traffic light': 7,
    'traffic sign': 8,
    'train': 9
}

INV_CLASS_MAP = {v: k for k, v in DEFAULT_CLASS_MAP.items()}
VALID_IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def load_class_map_from_yaml(yaml_path: str) -> tuple[dict, dict, list]:
    """Load class mapping and names from data.yaml if available."""
    if not os.path.exists(yaml_path):
        return DEFAULT_CLASS_MAP, INV_CLASS_MAP, list(DEFAULT_CLASS_MAP.keys())
    
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data_dict = yaml.safe_load(f)
    
    names = data_dict.get('names', [])
    if isinstance(names, list):
        class_map = {name: i for i, name in enumerate(names)}
        inv_class_map = {i: name for i, name in enumerate(names)}
        class_names = names
    elif isinstance(names, dict):
        class_map = {name: int(k) for k, name in names.items()}
        inv_class_map = {int(k): name for k, name in names.items()}
        class_names = [inv_class_map[i] for i in sorted(inv_class_map.keys())]
    else:
        class_map = DEFAULT_CLASS_MAP
        inv_class_map = INV_CLASS_MAP
        class_names = list(DEFAULT_CLASS_MAP.keys())
        
    return class_map, inv_class_map, class_names


def create_directories(target_path: str, nclients: int, yolo_structure: bool = False) -> None:
    """Create directory structure for server and clients."""
    os.makedirs(target_path, exist_ok=True)
    
    # Server directories
    os.makedirs(os.path.join(target_path, 'server', 'images'), exist_ok=True)
    os.makedirs(os.path.join(target_path, 'server', 'labels'), exist_ok=True)
    
    # Client directories
    for k in range(1, nclients + 1):
        client_dir = os.path.join(target_path, f'client{k}')
        if yolo_structure:
            os.makedirs(os.path.join(client_dir, 'train', 'images'), exist_ok=True)
            os.makedirs(os.path.join(client_dir, 'train', 'labels'), exist_ok=True)
            os.makedirs(os.path.join(client_dir, 'valid', 'images'), exist_ok=True)
            os.makedirs(os.path.join(client_dir, 'valid', 'labels'), exist_ok=True)
        else:
            os.makedirs(os.path.join(client_dir, 'images'), exist_ok=True)
            os.makedirs(os.path.join(client_dir, 'labels'), exist_ok=True)


def archive_directories(target_path: str, nclients: int) -> None:
    """Archive server and client directories into .tar files."""
    print('Archiving directories...')
    # Archive server
    server_path = os.path.join(target_path, 'server')
    if os.path.exists(server_path):
        tar_file_name = os.path.join(target_path, 'server.tar')
        with tarfile.open(tar_file_name, 'w') as tar_handle:
            tar_handle.add(server_path, arcname='server')
            
    # Archive clients
    for k in range(1, nclients + 1):
        client_path = os.path.join(target_path, f'client{k}')
        if os.path.exists(client_path):
            tar_file_name = os.path.join(target_path, f'client{k}.tar')
            with tarfile.open(tar_file_name, 'w') as tar_handle:
                tar_handle.add(client_path, arcname=f'client{k}')
    print('Archiving completed.')


def get_distribution_dataframe(class_names: list, nclients: int) -> pd.DataFrame:
    """Create a DataFrame to store the object count distribution across nodes."""
    columns = ['server'] + [f'client{k}' for k in range(1, nclients + 1)]
    index = ['Samples'] + class_names
    objects_distribution = pd.DataFrame(0, index=index, columns=columns)
    return objects_distribution


def get_labels_from_bdd100k(label_path: str, image_bases: list, class_map: dict, n_classes: int) -> dict:
    """
    Extract dominant class for each image from BDD100K label files.
    Identical logic to get_labels_from_kitti in prepare_kitti.py.
    """
    image_labels = {}
    print("Extracting dominant classes from labels...")
    
    for base_name in tqdm(image_bases, desc="Reading labels"):
        label_file = f'{base_name}.txt'
        label_filepath = os.path.join(label_path, label_file)
        class_counts = defaultdict(int)
        
        if os.path.exists(label_filepath):
            with open(label_filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    obj = parts[0]
                    if obj.isdigit():
                        c_id = int(obj)
                        if 0 <= c_id < n_classes:
                            class_counts[c_id] += 1
                    elif obj in class_map:
                        class_counts[class_map[obj]] += 1
        
        # Assign class with most objects, or random class if empty
        if class_counts:
            image_labels[base_name] = max(class_counts, key=class_counts.get)
        else:
            image_labels[base_name] = random.choice(list(range(n_classes)))
            
    return image_labels


def get_iid_splits(image_bases: list, nclients: int, val_frac: float) -> dict:
    """
    Return a dictionary to store IID and balanced mapping of BDD100K data split.
    Identical to get_iid_splits in prepare_kitti.py.
    If val_frac == 0, 100% of data is distributed evenly among clients.
    """
    print('Creating IID splits...')
    random.seed(0)
    total_size = len(image_bases)
    client_frac = (1.0 - val_frac) / nclients
    indices = list(range(total_size))
    client_split_size = int(total_size * client_frac)
    splits = {}
    
    # Create the client splits
    for k in range(1, nclients + 1):
        # For the last client when val_frac is 0, assign all remaining samples to avoid rounding drops
        if k == nclients and val_frac == 0:
            client_data = list(indices)
        else:
            client_data = random.sample(indices, min(client_split_size, len(indices)))
            
        s_client_data = set(client_data)
        for index in client_data:
            splits[image_bases[index]] = f'client{k}'
        indices = [x for x in indices if x not in s_client_data]
        
    # Create the server split (validation) if any remain
    for index in indices:
        splits[image_bases[index]] = 'server'
        
    return splits


def get_dirichlet_splits(label_path: str, image_bases: list, class_map: dict, inv_class_map: dict,
                         nclients: int, val_frac: float, alpha: float) -> dict:
    """
    Chia dữ liệu theo phân phối Dirichlet (Non-IID).
    Identical to get_dirichlet_splits in prepare_kitti.py.
    If val_frac == 0, 100% of training data is partitioned among clients.
    
    Args:
        label_path: đường dẫn đến labels BDD100K
        image_bases: danh sách tên gốc của ảnh (không kèm đuôi mở rộng)
        class_map: mapping từ tên class sang ID
        inv_class_map: mapping từ ID sang tên class
        nclients: số lượng clients
        val_frac: tỷ lệ validation cho server (nếu = 0 thì không tách tập val từ train)
        alpha: tham số Dirichlet (càng nhỏ càng Non-IID)
    
    Returns:
        dict: {image_base: 'client1'/'client2'/.../'server'}
    """
    print(f'Creating Dirichlet Non-IID splits with alpha={alpha} (val_frac={val_frac})...')
    random.seed(0)
    np.random.seed(0)
    n_classes = len(class_map)
    
    # Bước 1: Lấy labels cho tất cả ảnh
    image_labels = get_labels_from_bdd100k(label_path, image_bases, class_map, n_classes)
    
    # Bước 2: Nhóm ảnh theo class
    class_indices = defaultdict(list)
    for base_name, label in image_labels.items():
        class_indices[label].append(base_name)
        
    # Shuffle
    for label in class_indices:
        random.shuffle(class_indices[label])
        
    # Bước 3: Tách validation set (nếu val_frac > 0)
    val_indices = []
    train_class_indices = defaultdict(list)
    
    for class_id in range(n_classes):
        indices = class_indices[class_id]
        if val_frac > 0:
            n_val = int(len(indices) * val_frac)
            val_indices.extend(indices[:n_val])
            train_class_indices[class_id] = indices[n_val:]
        else:
            train_class_indices[class_id] = indices[:]
        
    # Bước 4: Chia training data theo Dirichlet
    client_indices = [[] for _ in range(nclients)]
    
    for class_id in range(n_classes):
        indices = train_class_indices[class_id]
        if len(indices) == 0:
            continue
            
        # Sinh phân phối Dirichlet
        proportions = np.random.dirichlet([alpha] * nclients)
        proportions = proportions / proportions.sum()
        splits_idx = (np.cumsum(proportions) * len(indices)).astype(int)[:-1]
        
        # Gán cho clients
        for client_id, idx_subset in enumerate(np.split(indices, splits_idx)):
            client_indices[client_id].extend(idx_subset.tolist() if isinstance(idx_subset, np.ndarray) else idx_subset)
            
    # Bước 5: Tạo dict kết quả
    splits = {}
    for client_id, indices in enumerate(client_indices):
        for idx in indices:
            splits[idx] = f'client{client_id + 1}'
            
    for idx in val_indices:
        splits[idx] = 'server'
        
    # In thống kê
    print(f'\n{"="*60}')
    print('DATA DISTRIBUTION SUMMARY')
    print(f'{"="*60}')
    print(f'Server (validation split from train): {len(val_indices)} samples')
    for i, indices in enumerate(client_indices):
        print(f'Client {i+1}: {len(indices)} samples')
        
    # In phân phối classes
    print(f'\n{"="*60}')
    print('CLASS DISTRIBUTION PER CLIENT')
    print(f'{"="*60}')
    
    for client_id in range(nclients):
        client_imgs = client_indices[client_id]
        class_dist = defaultdict(int)
        for img_id in client_imgs:
            class_dist[image_labels[img_id]] += 1
            
        print(f'\nClient {client_id + 1} ({len(client_imgs)} samples):')
        for class_id, count in sorted(class_dist.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(client_imgs)) * 100 if len(client_imgs) > 0 else 0
            print(f'  {inv_class_map.get(class_id, f"class_{class_id}"):20s}: {count:5d} ({percentage:5.1f}%)')
            
    return splits


def create_yaml_files(target_path: str, nclients: int, class_names: list, yolo_structure: bool,
                      external_val_img_path: str = None) -> None:
    """Generate data.yaml config files for server and all clients."""
    nc = len(class_names)
    names_dict = {i: name for i, name in enumerate(class_names)}
    
    # Server data.yaml
    server_dir = os.path.abspath(os.path.join(target_path, 'server'))
    server_val = 'images' if not external_val_img_path else os.path.abspath(external_val_img_path)
    server_yaml = {
        'path': server_dir,
        'train': 'images',
        'val': server_val,
        'nc': nc,
        'names': names_dict
    }
    with open(os.path.join(server_dir, 'data.yaml'), 'w', encoding='utf-8') as f:
        yaml.dump(server_yaml, f, sort_keys=False)
        
    # Client data.yamls
    for k in range(1, nclients + 1):
        client_dir = os.path.abspath(os.path.join(target_path, f'client{k}'))
        if yolo_structure:
            client_yaml = {
                'path': client_dir,
                'train': 'train/images',
                'val': 'valid/images',
                'nc': nc,
                'names': names_dict
            }
        else:
            val_path = os.path.abspath(os.path.join(server_dir, 'images')) if not external_val_img_path else os.path.abspath(external_val_img_path)
            client_yaml = {
                'path': client_dir,
                'train': 'images',
                'val': val_path,
                'nc': nc,
                'names': names_dict
            }
        with open(os.path.join(client_dir, 'data.yaml'), 'w', encoding='utf-8') as f:
            yaml.dump(client_yaml, f, sort_keys=False)
            
    print(f'Generated data.yaml config files for server and {nclients} clients.')


def process_bdd100k(img_path: str, label_path: str, target_path: str, data_yaml_path: str,
                     class_map: dict, nclients: int, val_frac: float, partition: str,
                     alpha: float, tar: bool, yolo_structure: bool = False, symlink: bool = False,
                     val_img_path: str = None, val_label_path: str = None) -> None:
    """
    Convert / split BDD100K annotations and data among server and clients.
    Identical workflow to process_kitti in prepare_kitti.py.
    """
    print(f'\n{"="*60}')
    print('BDD100K FEDERATED DATASET PREPARATION & PARTITIONING')
    print(f'{"="*60}')
    print(f'Train Image path  : {img_path}')
    print(f'Train Label path  : {label_path}')
    print(f'Target path       : {target_path}')
    print(f'Number of clients : {nclients}')
    print(f'Validation frac   : {val_frac}')
    if val_frac == 0 and val_img_path:
        print(f'Existing Val Path : {val_img_path}')
    print(f'Partition strategy: {partition}')
    if partition == 'dirichlet':
        print(f'Dirichlet alpha   : {alpha}')
    print(f'YOLO structure    : {yolo_structure} ({"train/valid subfolders" if yolo_structure else "server + client folders"})')
    print(f'Use symlink       : {symlink}')
    print(f'{"="*60}\n')
    
    # 1. Load class mapping
    class_map, inv_class_map, class_names = load_class_map_from_yaml(data_yaml_path)
    n_classes = len(class_names)
    
    # 2. Gather image files
    all_files = os.listdir(img_path)
    image_file_map = {}  # base_name -> full filename
    for f in all_files:
        ext = os.path.splitext(f)[1].lower()
        if ext in VALID_IMG_EXTENSIONS:
            base = os.path.splitext(f)[0]
            image_file_map[base] = f
            
    image_bases = sorted(list(image_file_map.keys()))
    print(f'Found {len(image_bases)} images in {img_path}')
    if len(image_bases) == 0:
        raise FileNotFoundError(f'No images found in {img_path}')
        
    # 3. Create target directories
    create_directories(target_path, nclients, yolo_structure=yolo_structure)
    
    # 4. Partition dataset (IID or Dirichlet Non-IID)
    if partition == 'iid':
        splits = get_iid_splits(image_bases, nclients, val_frac)
    elif partition == 'dirichlet':
        splits = get_dirichlet_splits(label_path, image_bases, class_map, inv_class_map,
                                      nclients, val_frac, alpha)
    else:
        raise ValueError(f'Unknown partition strategy: {partition}. Choose "iid" or "dirichlet"')
        
    # 5. Initialize distribution tracking DataFrame
    objects_distribution = get_distribution_dataframe(class_names, nclients)
    
    # 6. Copy / link files and record object distribution
    print('\nDistributing images and labels...')
    
    # Cache server files if yolo_structure is requested (to populate valid folders for clients)
    server_samples = []
    
    for base in tqdm(image_bases, desc="Processing samples"):
        destination = splits[base]
        img_filename = image_file_map[base]
        lbl_filename = f'{base}.txt'
        
        src_img = os.path.join(img_path, img_filename)
        src_lbl = os.path.join(label_path, lbl_filename)
        
        objects_distribution.loc['Samples', destination] += 1
        
        # Determine target folders
        if destination == 'server':
            dst_img_dir = os.path.join(target_path, 'server', 'images')
            dst_lbl_dir = os.path.join(target_path, 'server', 'labels')
            server_samples.append((img_filename, lbl_filename))
        else:
            if yolo_structure:
                dst_img_dir = os.path.join(target_path, destination, 'train', 'images')
                dst_lbl_dir = os.path.join(target_path, destination, 'train', 'labels')
            else:
                dst_img_dir = os.path.join(target_path, destination, 'images')
                dst_lbl_dir = os.path.join(target_path, destination, 'labels')
                
        dst_img = os.path.join(dst_img_dir, img_filename)
        dst_lbl = os.path.join(dst_lbl_dir, lbl_filename)
        
        # Copy / symlink image
        if symlink:
            if os.path.exists(dst_img):
                os.remove(dst_img)
            os.symlink(os.path.abspath(src_img), dst_img)
        else:
            shutil.copyfile(src_img, dst_img)
            
        # Copy / process label and count annotations
        if os.path.exists(src_lbl):
            with open(src_lbl, 'r', encoding='utf-8') as f_in, open(dst_lbl, 'w', encoding='utf-8') as f_out:
                for line in f_in:
                    line_s = line.strip()
                    if not line_s:
                        continue
                    parts = line_s.split()
                    obj = parts[0]
                    if obj.isdigit():
                        c_id = int(obj)
                        if 0 <= c_id < n_classes:
                            f_out.write(line_s + '\n')
                            c_name = inv_class_map.get(c_id, f'class_{c_id}')
                            if c_name in objects_distribution.index:
                                objects_distribution.loc[c_name, destination] += 1
                    elif obj in class_map:
                        c_id = class_map[obj]
                        # Write converted class id format if necessary
                        f_out.write(f'{c_id} ' + ' '.join(parts[1:]) + '\n')
                        if obj in objects_distribution.index:
                            objects_distribution.loc[obj, destination] += 1
        else:
            # Create empty label file if none exists
            open(dst_lbl, 'w', encoding='utf-8').close()
            
    # 7. Populate valid/ for each client with validation set
    if yolo_structure:
        # Case A: val_frac == 0 and user provided existing val folder
        if val_frac == 0 and val_img_path and os.path.exists(val_img_path):
            print(f'\nPopulating external validation set from {val_img_path} into clients...')
            val_files = [f for f in os.listdir(val_img_path) if os.path.splitext(f)[1].lower() in VALID_IMG_EXTENSIONS]
            for k in range(1, nclients + 1):
                client_val_img_dir = os.path.join(target_path, f'client{k}', 'valid', 'images')
                client_val_lbl_dir = os.path.join(target_path, f'client{k}', 'valid', 'labels')
                for f in val_files:
                    base = os.path.splitext(f)[0]
                    s_img = os.path.join(val_img_path, f)
                    s_lbl = os.path.join(val_label_path, f'{base}.txt') if val_label_path else ""
                    d_img = os.path.join(client_val_img_dir, f)
                    d_lbl = os.path.join(client_val_lbl_dir, f'{base}.txt')
                    if symlink:
                        if os.path.exists(d_img): os.remove(d_img)
                        os.symlink(os.path.abspath(s_img), d_img)
                        if os.path.exists(s_lbl):
                            if os.path.exists(d_lbl): os.remove(d_lbl)
                            os.symlink(os.path.abspath(s_lbl), d_lbl)
                    else:
                        shutil.copyfile(s_img, d_img)
                        if os.path.exists(s_lbl):
                            shutil.copyfile(s_lbl, d_lbl)
                        else:
                            open(d_lbl, 'w', encoding='utf-8').close()
        # Case B: Validation set was split from train (val_frac > 0)
        elif server_samples:
            print('\nPopulating validation set (server) into client valid/ directories...')
            src_srv_img_dir = os.path.join(target_path, 'server', 'images')
            src_srv_lbl_dir = os.path.join(target_path, 'server', 'labels')
            
            for k in range(1, nclients + 1):
                client_val_img_dir = os.path.join(target_path, f'client{k}', 'valid', 'images')
                client_val_lbl_dir = os.path.join(target_path, f'client{k}', 'valid', 'labels')
                
                for img_fn, lbl_fn in server_samples:
                    s_img = os.path.join(src_srv_img_dir, img_fn)
                    s_lbl = os.path.join(src_srv_lbl_dir, lbl_fn)
                    d_img = os.path.join(client_val_img_dir, img_fn)
                    d_lbl = os.path.join(client_val_lbl_dir, lbl_fn)
                    
                    if symlink:
                        if os.path.exists(d_img):
                            os.remove(d_img)
                        os.symlink(os.path.abspath(s_img), d_img)
                        if os.path.exists(d_lbl):
                            os.remove(d_lbl)
                        os.symlink(os.path.abspath(s_lbl), d_lbl)
                    else:
                        shutil.copyfile(s_img, d_img)
                        if os.path.exists(s_lbl):
                            shutil.copyfile(s_lbl, d_lbl)
                        else:
                            open(d_lbl, 'w', encoding='utf-8').close()
                            
    # 8. Save distribution dataframe to CSV
    csv_path = os.path.join(target_path, 'objects_distribution.csv')
    objects_distribution.to_csv(csv_path)
    print(f'\nObjects distribution saved to {csv_path}')
    
    print(f'\n{"="*60}')
    print('OBJECTS DISTRIBUTION TABLE')
    print(f'{"="*60}')
    print(objects_distribution.to_string())
    
    # 9. Create data.yaml files
    external_val = val_img_path if (val_frac == 0 and val_img_path and not yolo_structure) else None
    create_yaml_files(target_path, nclients, class_names, yolo_structure=yolo_structure, external_val_img_path=external_val)
    
    # 10. Archive directories if requested
    if tar:
        archive_directories(target_path, nclients)
        
    print(f'\n{"="*60}')
    print('DATASET PARTITIONING COMPLETED SUCCESSFULLY!')
    print(f'Target directory: {os.path.abspath(target_path)}')
    print(f'{"="*60}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Partition BDD100K Dataset for Federated Learning')
    
    parser.add_argument('--img-path', type=str,
                        default='/media/data3/home/truongduy/BDD100K/train/images',
                        help='Path to input images directory')
    parser.add_argument('--label-path', type=str,
                        default='/media/data3/home/truongduy/BDD100K/train/labels',
                        help='Path to input labels directory')
    parser.add_argument('--val-img-path', type=str,
                        default='/media/data3/home/truongduy/BDD100K/val/images',
                        help='Path to existing validation images (used when val-frac is 0)')
    parser.add_argument('--val-label-path', type=str,
                        default='/media/data3/home/truongduy/BDD100K/val/labels',
                        help='Path to existing validation labels (used when val-frac is 0)')
    parser.add_argument('--target-path', type=str,
                        default='Dataset/BDD100K_10clients_noniid',
                        help='Path to target directory for output partitions')
    parser.add_argument('--data', type=str,
                        default='/media/data3/home/truongduy/BDD100K/data.yaml',
                        help='Path to dataset data.yaml file')
    parser.add_argument('--nclients', type=int, default=10,
                        help='Number of clients in federated experiment (default: 10)')
    parser.add_argument('--val-frac', type=float, default=0.0,
                        help='Fraction of data held by the server for validation (default: 0.0 if already have val set)')
    parser.add_argument('--partition', type=str, default='dirichlet', choices=['iid', 'dirichlet'],
                        help='Data partitioning strategy: iid or dirichlet (default: dirichlet)')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Dirichlet alpha parameter - smaller = more non-IID (default: 0.5)')
    parser.add_argument('--tar', action='store_true',
                        help='Archive the directories of the federated participants into .tar files')
    parser.add_argument('--yolo-structure', action='store_true', default=True,
                        help='Organize client folders into train/ and valid/ subfolders with data.yaml (default: True)')
    parser.add_argument('--no-yolo-structure', dest='yolo_structure', action='store_false',
                        help='Organize directly as server/ and clientX/ with images/ and labels/ subfolders (prepare_kitti format)')
    parser.add_argument('--symlink', action='store_true',
                        help='Create symbolic links instead of copying image files to save disk space')
    
    args = parser.parse_args()
    
    process_bdd100k(
        img_path=args.img_path,
        label_path=args.label_path,
        target_path=args.target_path,
        data_yaml_path=args.data,
        class_map=DEFAULT_CLASS_MAP,
        nclients=args.nclients,
        val_frac=args.val_frac,
        partition=args.partition,
        alpha=args.alpha,
        tar=args.tar,
        yolo_structure=args.yolo_structure,
        symlink=args.symlink,
        val_img_path=args.val_img_path,
        val_label_path=args.val_label_path
    )

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fast Streaming Bucket Stratified Subsetting for BDD100K.
Method 1: Streaming Sampling with Early Stopping & Multithreaded I/O.
Extracts balanced 5,000 train & 2,000 val in seconds instead of scanning all 70,000 files.
"""

import argparse
import os
import random
import shutil
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import yaml

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


def read_dominant_class(img_file: str, label_dir: str, n_classes: int = 10) -> tuple[str, int]:
    """Read label file and return (img_file, dominant_class_id)."""
    base = os.path.splitext(img_file)[0]
    lbl_file = os.path.join(label_dir, f"{base}.txt")
    class_counts = defaultdict(int)
    
    if os.path.exists(lbl_file):
        try:
            with open(lbl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    c_id = int(parts[0]) if parts[0].isdigit() else DEFAULT_CLASS_MAP.get(parts[0], -1)
                    if 0 <= c_id < n_classes:
                        class_counts[c_id] += 1
        except Exception:
            pass
            
    if class_counts:
        dom_class = max(class_counts, key=class_counts.get)
    else:
        dom_class = random.randint(0, n_classes - 1)
        
    return img_file, dom_class


def sample_streaming_buckets(img_dir: str, label_dir: str, target_size: int,
                             n_classes: int = 10, batch_size: int = 500,
                             max_workers: int = 32, seed: int = 42) -> list:
    """
    Streaming stratified bucket sampling with early stopping.
    Reads images in batches, fills per-class buckets, and stops as soon as targets are met.
    """
    random.seed(seed)
    all_images = [f for f in os.listdir(img_dir) if os.path.splitext(f)[1].lower() in VALID_IMG_EXTENSIONS]
    random.shuffle(all_images)
    
    per_class_target = target_size // n_classes
    buckets = defaultdict(list)
    overflow_pool = []
    scanned = 0
    total_available = len(all_images)
    
    print(f"\n[Sampling {os.path.basename(os.path.dirname(img_dir))}] Target: {target_size} total ({per_class_target}/class)")
    pbar = tqdm(total=target_size, desc="Collecting samples")
    
    # Process in batches with multithreading
    for i in range(0, total_available, batch_size):
        batch = all_images[i:i + batch_size]
        scanned += len(batch)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(read_dominant_class, img, label_dir, n_classes) for img in batch]
            results = [f.result() for f in futures]
            
        for img_file, c_id in results:
            if len(buckets[c_id]) < per_class_target:
                buckets[c_id].append(img_file)
                pbar.update(1)
            else:
                overflow_pool.append(img_file)
                
        total_in_buckets = sum(len(b) for b in buckets.values())
        
        # Early stopping condition 1: All 10 class buckets are perfectly filled
        if total_in_buckets >= target_size:
            print(f"\n⚡ Perfect balance achieved! All {n_classes} classes reached {per_class_target} samples.")
            break
            
        # Early stopping condition 2: Non-rare classes are full, and total collected + overflow can satisfy target
        # Stop after scanning enough to capture rare classes without scanning all 70k files
        if scanned >= max(target_size * 2, 7000) and (total_in_buckets + len(overflow_pool) >= target_size):
            # Check if major classes are full
            full_classes = sum(1 for c in range(n_classes) if len(buckets[c]) >= per_class_target)
            if full_classes >= 7 or scanned >= 15000:
                print(f"\n⚡ Early stopping threshold reached ({full_classes}/{n_classes} buckets filled).")
                break
                
    pbar.close()
    
    # Assemble final selected list
    selected = []
    for c in range(n_classes):
        selected.extend(buckets[c])
        
    needed = target_size - len(selected)
    if needed > 0 and overflow_pool:
        random.shuffle(overflow_pool)
        selected.extend(overflow_pool[:needed])
        
    random.shuffle(selected)
    
    # Print class distribution statistics
    print(f"\n{'='*55}")
    print(f"Scanned {scanned}/{total_available} files ({scanned/total_available*100:.1f}%) -> Selected {len(selected)} samples")
    print(f"{'Class ID':<10}{'Class Name':<18}{'Count':<10}{'Percentage':<10}")
    print(f"{'-'*55}")
    
    class_counts = defaultdict(int)
    for c, img_list in buckets.items():
        class_counts[c] = len(img_list)
        
    for c in range(n_classes):
        cnt = class_counts[c]
        pct = (cnt / len(selected)) * 100 if selected else 0
        print(f"{c:<10}{INV_CLASS_MAP[c]:<18}{cnt:<10}{pct:.1f}%")
    print(f"{'='*55}\n")
    
    return selected


def copy_subset(selected_images: list, src_img_dir: str, src_lbl_dir: str,
                dst_img_dir: str, dst_lbl_dir: str, symlink: bool = True):
    """Copy or symlink selected images and labels."""
    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_lbl_dir, exist_ok=True)
    
    for img_file in tqdm(selected_images, desc=f"Writing to {os.path.basename(os.path.dirname(dst_img_dir))}"):
        base = os.path.splitext(img_file)[0]
        s_img = os.path.join(src_img_dir, img_file)
        s_lbl = os.path.join(src_lbl_dir, f"{base}.txt")
        d_img = os.path.join(dst_img_dir, img_file)
        d_lbl = os.path.join(dst_lbl_dir, f"{base}.txt")
        
        if symlink:
            if os.path.exists(d_img): os.remove(d_img)
            os.symlink(os.path.abspath(s_img), d_img)
            if os.path.exists(s_lbl):
                if os.path.exists(d_lbl): os.remove(d_lbl)
                os.symlink(os.path.abspath(s_lbl), d_lbl)
            else:
                open(d_lbl, 'w', encoding='utf-8').close()
        else:
            shutil.copyfile(s_img, d_img)
            if os.path.exists(s_lbl):
                shutil.copyfile(s_lbl, d_lbl)
            else:
                open(d_lbl, 'w', encoding='utf-8').close()


def create_subset(bdd_root: str, output_root: str, n_train: int = 5000, n_val: int = 2000,
                  symlink: bool = True, seed: int = 42, max_workers: int = 32):
    """Create subset for BDD100K with specified train/val sizes."""
    t0 = time.time()
    print(f"\n{'='*60}")
    print(f"FAST BDD100K SUBSETTING (Method 1: Streaming Buckets)")
    print(f"Source       : {bdd_root}")
    print(f"Output       : {output_root}")
    print(f"Target Train : {n_train} images")
    print(f"Target Val   : {n_val} images")
    print(f"Symlink      : {symlink}")
    print(f"{'='*60}")
    
    # 1. Train sampling
    train_src_img = os.path.join(bdd_root, 'train', 'images')
    train_src_lbl = os.path.join(bdd_root, 'train', 'labels')
    selected_train = sample_streaming_buckets(train_src_img, train_src_lbl, target_size=n_train,
                                              n_classes=10, seed=seed, max_workers=max_workers)
    
    # 2. Val sampling
    val_src_img = os.path.join(bdd_root, 'val', 'images')
    if not os.path.exists(val_src_img):
        val_src_img = os.path.join(bdd_root, 'valid', 'images')
    val_src_lbl = os.path.join(bdd_root, 'val', 'labels')
    if not os.path.exists(val_src_lbl):
        val_src_lbl = os.path.join(bdd_root, 'valid', 'labels')
        
    selected_val = sample_streaming_buckets(val_src_img, val_src_lbl, target_size=n_val,
                                            n_classes=10, seed=seed, max_workers=max_workers)
    
    # 3. Write files
    copy_subset(selected_train, train_src_img, train_src_lbl,
                os.path.join(output_root, 'train', 'images'),
                os.path.join(output_root, 'train', 'labels'),
                symlink=symlink)
                
    copy_subset(selected_val, val_src_img, val_src_lbl,
                os.path.join(output_root, 'val', 'images'),
                os.path.join(output_root, 'val', 'labels'),
                symlink=symlink)
                
    # 4. Create data.yaml
    yaml_dict = {
        'path': os.path.abspath(output_root),
        'train': 'train/images',
        'val': 'val/images',
        'nc': 10,
        'names': {i: name for i, name in enumerate(DEFAULT_CLASS_MAP.keys())}
    }
    with open(os.path.join(output_root, 'data.yaml'), 'w', encoding='utf-8') as f:
        yaml.dump(yaml_dict, f, sort_keys=False)
        
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"SUBSET CREATED SUCCESSFULLY in {elapsed:.2f} seconds!")
    print(f"Train: {len(selected_train)} images -> {os.path.join(output_root, 'train')}")
    print(f"Val  : {len(selected_val)} images -> {os.path.join(output_root, 'val')}")
    print(f"Yaml : {os.path.join(output_root, 'data.yaml')}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create balanced subset of BDD100K via Streaming Buckets')
    parser.add_argument('--bdd_root', type=str, default='/media/data3/home/truongduy/BDD100K',
                        help='Path to full BDD100K dataset root')
    parser.add_argument('--output_root', type=str, default='/media/data3/home/truongduy/BDD100K_subset_5k_2k',
                        help='Path to output subset directory')
    parser.add_argument('--n_train', type=int, default=5000, help='Number of training images (default: 5000)')
    parser.add_argument('--n_val', type=int, default=2000, help='Number of validation images (default: 2000)')
    parser.add_argument('--copy', action='store_true', help='Copy files instead of symlink')
    parser.add_argument('--workers', type=int, default=32, help='Number of worker threads (default: 32)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    create_subset(
        bdd_root=args.bdd_root,
        output_root=args.output_root,
        n_train=args.n_train,
        n_val=args.n_val,
        symlink=not args.copy,
        seed=args.seed,
        max_workers=args.workers
    )

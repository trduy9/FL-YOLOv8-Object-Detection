#!/usr/bin/env python3
"""
partition_kitti_iid_symlink.py

Script chia dataset KITTI theo dạng IID cho N clients sử dụng Symbolic Link (Symlink).
- Không nhân bản file ảnh -> Tốn 0 MB dung lượng ổ cứng.
- Tốc độ cực nhanh (hoàn thành trong vài giây).
- Hỗ trợ cả 2 chế độ IID:
    1. 'stratified' (khuyến nghị): Chia đều cả số lượng ảnh và phân phối từng class cho mỗi client.
    2. 'random': Chia ngẫu nhiên đồng đều số lượng ảnh.
- Tự động tạo thư mục chuẩn YOLO và sinh file data.yaml cho từng client.
"""

import os
import sys
import glob
import random
import argparse
import yaml
import csv
from collections import defaultdict

DEFAULT_CLASS_NAMES = [
    'Car', 'Van', 'Truck', 'Pedestrian', 
    'Person_sitting', 'Cyclist', 'Tram', 'Misc'
]

def get_dominant_class(label_path):
    """Đọc nhãn YOLO và trả về class xuất hiện nhiều nhất trong ảnh."""
    if not os.path.exists(label_path):
        return -1
    counts = defaultdict(int)
    try:
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    counts[int(parts[0])] += 1
    except Exception:
        return -1
    if not counts:
        return -1
    return max(counts, key=counts.get)


def collect_images_and_labels(source_dir, clean_labels_dir=None):
    """
    Thu thập tất cả các cặp (ảnh, nhãn) từ source_dir.
    Có thể đọc từ 1 thư mục chứa ảnh hoặc từ thư mục dataset cũ (client1, client2...).
    """
    exts = ('*.png', '*.jpg', '*.jpeg')
    image_paths = []
    
    # 1. Quét tìm tất cả file ảnh
    for ext in exts:
        image_paths.extend(glob.glob(os.path.join(source_dir, "**", ext), recursive=True))
    
    # Lọc bỏ ảnh nằm trong thư mục validation nếu quét đệ quy
    image_paths = [p for p in image_paths if "valid" not in p and "val" not in p]
    image_paths = sorted(list(set(image_paths)))
    
    pairs = []
    for img_p in image_paths:
        base_name = os.path.splitext(os.path.basename(img_p))[0]
        
        # Tìm file label tương ứng
        if clean_labels_dir and os.path.isdir(clean_labels_dir):
            lbl_p = os.path.join(clean_labels_dir, f"{base_name}.txt")
        else:
            # Thử tìm cạnh thư mục images
            parent = os.path.dirname(os.path.dirname(img_p))
            lbl_p = os.path.join(parent, "labels", f"{base_name}.txt")
            if not os.path.exists(lbl_p):
                lbl_p = os.path.join(os.path.dirname(img_p), f"{base_name}.txt")
                
        if os.path.exists(lbl_p):
            pairs.append((os.path.abspath(img_p), os.path.abspath(lbl_p)))
        else:
            print(f"Warning: Không tìm thấy nhãn cho {img_p}")
            
    return pairs


def split_iid_stratified(pairs, num_clients, seed=42):
    """Chia IID có cân bằng phân phối class (Stratified IID)."""
    random.seed(seed)
    class_to_pairs = defaultdict(list)
    
    for img_p, lbl_p in pairs:
        dom_cls = get_dominant_class(lbl_p)
        class_to_pairs[dom_cls].append((img_p, lbl_p))
        
    client_splits = [[] for _ in range(num_clients)]
    
    # Với mỗi class, xáo trộn rồi chia đều xoay vòng (round-robin) cho các client
    for cls_id in sorted(class_to_pairs.keys()):
        cls_pairs = class_to_pairs[cls_id]
        random.shuffle(cls_pairs)
        for i, item in enumerate(cls_pairs):
            client_splits[i % num_clients].append(item)
            
    return client_splits


def split_iid_random(pairs, num_clients, seed=42):
    """Chia IID ngẫu nhiên thuần túy (Random IID)."""
    random.seed(seed)
    shuffled = pairs.copy()
    random.shuffle(shuffled)
    
    client_splits = [[] for _ in range(num_clients)]
    for i, item in enumerate(shuffled):
        client_splits[i % num_clients].append(item)
    return client_splits


def create_symlink_dataset(client_splits, output_root, val_path, class_names):
    """Tạo các thư mục client bằng Symlink và ghi data.yaml."""
    os.makedirs(output_root, exist_ok=True)
    abs_val_path = os.path.abspath(val_path) if val_path else None
    
    print("\n" + "="*60)
    print(f"🚀 BẮT ĐẦU TẠO DATASET SYMLINK TẠI: {output_root}")
    print("="*60)
    
    for client_idx, pairs in enumerate(client_splits):
        client_name = f"client{client_idx + 1}"
        client_dir = os.path.join(output_root, client_name)
        train_img_dir = os.path.join(client_dir, "train", "images")
        train_lbl_dir = os.path.join(client_dir, "train", "labels")
        
        os.makedirs(train_img_dir, exist_ok=True)
        os.makedirs(train_lbl_dir, exist_ok=True)
        
        # 1. Tạo symlink cho từng ảnh và nhãn
        for img_p, lbl_p in pairs:
            img_target = os.path.join(train_img_dir, os.path.basename(img_p))
            lbl_target = os.path.join(train_lbl_dir, os.path.basename(lbl_p))
            
            if os.path.islink(img_target) or os.path.exists(img_target):
                os.remove(img_target)
            if os.path.islink(lbl_target) or os.path.exists(lbl_target):
                os.remove(lbl_target)
                
            os.symlink(img_p, img_target)
            os.symlink(lbl_p, lbl_target)
            
        # 2. Tạo symlink cho thư mục validation nếu có
        val_target_str = "valid/images"
        if abs_val_path and os.path.exists(abs_val_path):
            client_valid_dir = os.path.join(client_dir, "valid")
            if os.path.islink(client_valid_dir) or os.path.exists(client_valid_dir):
                if os.path.islink(client_valid_dir):
                    os.remove(client_valid_dir)
            try:
                os.symlink(abs_val_path, client_valid_dir)
                val_target_str = "valid/images"
            except Exception:
                # Nếu không symlink được thư mục, gán trực tiếp đường dẫn tuyệt đối vào data.yaml
                val_target_str = os.path.join(abs_val_path, "images")
                
        # 3. Tạo data.yaml cho client
        data_yaml_path = os.path.join(client_dir, "data.yaml")
        yaml_content = {
            "path": os.path.abspath(client_dir),
            "train": "train/images",
            "val": val_target_str,
            "nc": len(class_names),
            "names": {i: name for i, name in enumerate(class_names)}
        }
        with open(data_yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_content, f, sort_keys=False)
            
        print(f"✅ {client_name}: Đã tạo symlink {len(pairs)} ảnh & nhãn (Dung lượng: ~0 MB)")


def save_objects_distribution_csv(client_splits, output_root, class_names, val_dir=None):
    """
    Tạo file objects_distribution.csv và object_distribution.csv
    thống kê chi tiết số lượng mẫu (Samples) và số lượng object bounding box
    của từng class cho từng client (và server/validation).
    """
    client_headers = [f"client{i+1}" for i in range(len(client_splits))]
    headers = [""]
    
    # 1. Thống kê tập server/validation (nếu có)
    server_samples = 0
    server_counts = defaultdict(int)
    has_server = False
    
    if val_dir:
        v_lbl_dir = os.path.join(val_dir, "labels") if os.path.isdir(os.path.join(val_dir, "labels")) else val_dir
        if os.path.isdir(v_lbl_dir):
            has_server = True
            headers.append("server")
            val_files = [os.path.join(v_lbl_dir, f) for f in os.listdir(v_lbl_dir) if f.endswith(".txt")]
            server_samples = len(val_files)
            for vf in val_files:
                try:
                    with open(vf, "r") as fp:
                        for line in fp:
                            parts = line.strip().split()
                            if parts:
                                cls_id = int(parts[0])
                                if 0 <= cls_id < len(class_names):
                                    server_counts[class_names[cls_id]] += 1
                except Exception:
                    pass
                    
    headers.extend(client_headers)
    headers.append("Total")
    
    # 2. Thống kê từng client
    client_samples = []
    client_class_counts = {c: [] for c in class_names}
    client_total_objects = []
    
    for client_idx, pairs in enumerate(client_splits):
        client_samples.append(len(pairs))
        c_counts = defaultdict(int)
        
        for _, lbl_p in pairs:
            if os.path.exists(lbl_p):
                try:
                    with open(lbl_p, "r") as fp:
                        for line in fp:
                            parts = line.strip().split()
                            if parts:
                                cls_id = int(parts[0])
                                if 0 <= cls_id < len(class_names):
                                    c_counts[class_names[cls_id]] += 1
                except Exception:
                    pass
                    
        total_objs = sum(c_counts.values())
        client_total_objects.append(total_objs)
        for cname in class_names:
            client_class_counts[cname].append(c_counts[cname])
            
    # 3. Tạo các hàng dữ liệu CSV
    csv_rows = []
    
    # Hàng 1: Samples
    sample_row = ["Samples"]
    if has_server:
        sample_row.append(server_samples)
    sample_row.extend(client_samples)
    total_samples_all = (server_samples if has_server else 0) + sum(client_samples)
    sample_row.append(total_samples_all)
    csv_rows.append(sample_row)
    
    # Các hàng tiếp theo: Từng class
    for cname in class_names:
        row = [cname]
        if has_server:
            row.append(server_counts[cname])
        row.extend(client_class_counts[cname])
        total_class_all = (server_counts[cname] if has_server else 0) + sum(client_class_counts[cname])
        row.append(total_class_all)
        csv_rows.append(row)
        
    # Hàng cuối: Total_Objects
    total_obj_row = ["Total_Objects"]
    server_tot_objs = sum(server_counts.values()) if has_server else 0
    if has_server:
        total_obj_row.append(server_tot_objs)
    total_obj_row.extend(client_total_objects)
    all_objs_sum = server_tot_objs + sum(client_total_objects)
    total_obj_row.append(all_objs_sum)
    csv_rows.append(total_obj_row)
    
    # Lưu thành cả 2 tên file để tương thích tuyệt đối
    csv_filenames = ["objects_distribution.csv", "object_distribution.csv"]
    for fname in csv_filenames:
        out_csv_path = os.path.join(output_root, fname)
        with open(out_csv_path, "w", newline="", encoding="utf-8") as fp:
            writer = csv.writer(fp)
            writer.writerow(headers)
            writer.writerows(csv_rows)
            
    print(f"📄 Đã tạo file: {os.path.join(output_root, 'objects_distribution.csv')}")
    print(f"📄 Đã tạo file: {os.path.join(output_root, 'object_distribution.csv')}")


def print_statistics(client_splits, class_names):
    """In bảng phân phối thống kê số lượng ảnh và class của từng client."""
    print("\n" + "="*70)
    print("📊 BẢNG PHÂN PHỐI DỮ LIỆU IID GIỮA CÁC CLIENTS")
    print("="*70)
    
    header = f"{'Client':<10} | {'Tổng mẫu':<10} | Class phân phối chủ đạo"
    print(header)
    print("-" * 70)
    
    for idx, pairs in enumerate(client_splits):
        cname = f"client{idx + 1}"
        cls_counts = defaultdict(int)
        for _, lbl_p in pairs:
            d_cls = get_dominant_class(lbl_p)
            if d_cls >= 0:
                cls_counts[d_cls] += 1
        
        counts_str = ", ".join([f"{class_names[c]}:{cls_counts[c]}" for c in sorted(cls_counts.keys())[:4]])
        print(f"{cname:<10} | {len(pairs):<10} | {counts_str}...")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Chia dataset KITTI theo dạng IID sử dụng Symlink (0 MB)")
    parser.add_argument("--source-dir", type=str, 
                        default="/media/data3/home/truongduy/FL-YOLOv8-Object-Detection/Data/Dataset/KITTI_10clients_noniid",
                        help="Thư mục chứa ảnh KITTI (có thể là folder dataset non-iid cũ hoặc folder ảnh gốc)")
    parser.add_argument("--clean-labels-dir", type=str, 
                        default="/media/data3/home/truongduy/label_kitti_yolo_clean",
                        help="Thư mục chứa nhãn gốc sạch của KITTI (mặc định: label_kitti_yolo_clean)")
    parser.add_argument("--val-dir", type=str, 
                        default="/media/data3/home/truongduy/KITTI_YOLO_training/valid",
                        help="Thư mục validation gốc của KITTI")
    parser.add_argument("--output-dir", type=str, 
                        default="/media/data3/home/truongduy/FL-YOLOv8-Object-Detection/Data/Dataset/KITTI_10clients_iid",
                        help="Thư mục đầu ra cho 10 clients IID")
    parser.add_argument("--nclients", type=int, default=10, 
                        help="Số lượng clients (mặc định: 10)")
    parser.add_argument("--mode", type=str, choices=['stratified', 'random'], default='stratified',
                        help="Chế độ chia: 'stratified' (cân bằng class, chuẩn IID) hoặc 'random'")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed để tái lập kết quả")
    
    args = parser.parse_args()
    
    print(f"🔍 Đang thu thập ảnh từ: {args.source_dir} ...")
    pairs = collect_images_and_labels(args.source_dir, args.clean_labels_dir)
    print(f"📦 Đã tìm thấy tổng cộng {len(pairs)} ảnh & nhãn hợp lệ.")
    
    if len(pairs) == 0:
        print("❌ Lỗi: Không tìm thấy ảnh nào. Vui lòng kiểm tra lại đường dẫn source-dir.")
        sys.exit(1)
        
    print(f"⚙️ Đang phân chia dữ liệu IID ({args.mode}) cho {args.nclients} clients...")
    if args.mode == 'stratified':
        splits = split_iid_stratified(pairs, args.nclients, seed=args.seed)
    else:
        splits = split_iid_random(pairs, args.nclients, seed=args.seed)
        
    create_symlink_dataset(splits, args.output_dir, args.val_dir, DEFAULT_CLASS_NAMES)
    save_objects_distribution_csv(splits, args.output_dir, DEFAULT_CLASS_NAMES, val_dir=args.val_dir)
    print_statistics(splits, DEFAULT_CLASS_NAMES)
    print(f"🎉 HOÀN TẤT! Bạn có thể sử dụng dataset tại: {args.output_dir}\n")

if __name__ == "__main__":
    main()

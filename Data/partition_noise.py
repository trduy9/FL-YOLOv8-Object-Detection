import os
import random
import shutil
import numpy as np
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inject noisy labels into partitioned FL clients."
    )

    parser.add_argument(
    "--clients_root",
    type=str,
    required=True,
    help=""
)

    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help=""
    )
    
    parser.add_argument(
        "--noisy_label_dir",
        type=str,
        required=True
)

    parser.add_argument(
        "--client_noise_ratio",
        type=float,
        default=0.5,
        help=""
    )

    parser.add_argument(
        "--sample_noise_ratio",
        type=float,
        default=0.5,
        help=""
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    parser.add_argument(
        "--backup_clean",
        action="store_true",
        help=""
    )

    return parser.parse_args()


def get_clients(clients_root):
    clients = []

    for d in sorted(os.listdir(clients_root)):
        path = os.path.join(clients_root, d)

        if os.path.isdir(path):
            clients.append(d)

    return clients


def backup_labels(label_dir):
    backup_dir = os.path.join(
        os.path.dirname(label_dir),
        "labels_clean_backup"
    )

    if os.path.exists(backup_dir):
        return

    shutil.copytree(label_dir, backup_dir)


def inject_noise(
        clients_root,
        noisy_label_dir,
        client_noise_ratio,
        sample_noise_ratio,
        seed=0,
        backup=False):

    random.seed(seed)

    clients = get_clients(clients_root)

    if len(clients) == 0:
        raise RuntimeError("Cant find clients")

    #######################################################
    #
    #######################################################

    num_noisy_clients = int(round(
        len(clients) * client_noise_ratio
    ))

    num_noisy_clients = max(1, num_noisy_clients)

    noisy_clients = random.sample(
        clients,
        num_noisy_clients
    )

    print("=" * 70)
    print("Selected noisy clients:")
    print(noisy_clients)
    print("=" * 70)

    #######################################################
    # 
    #######################################################

    total_replaced = 0

    for client in clients:

        label_dir = os.path.join(
            clients_root,
            client,
            "train",
            "labels"
        )

        if not os.path.isdir(label_dir):
            print(f"Skip {client}, label folder not found.")
            continue

        label_files = sorted([
            f for f in os.listdir(label_dir)
            if f.endswith(".txt")
        ])

        if backup:
            backup_labels(label_dir)

        if client not in noisy_clients:

            print(
                f"{client}: CLEAN "
                f"({len(label_files)} labels)"
            )

            continue

        ###################################################
        # c
        ###################################################

        num_noisy_samples = int(round(
            len(label_files) * sample_noise_ratio
        ))

        num_noisy_samples = max(1, num_noisy_samples)

        selected = random.sample(
            label_files,
            num_noisy_samples
        )

        replaced = 0

        for lbl in selected:

            noisy_path = os.path.join(
                noisy_label_dir,
                lbl
            )

            clean_path = os.path.join(
                label_dir,
                lbl
            )

            if not os.path.exists(noisy_path):
                print(f"Missing noisy label: {lbl}")
                continue

            shutil.copy2(
                noisy_path,
                clean_path
            )

            replaced += 1

        total_replaced += replaced

        print(
            f"{client}: "
            f"{replaced}/{len(label_files)} "
            f"labels replaced"
        )

    #######################################################
    # summary
    #######################################################

    print("\n")
    print("=" * 70)
    print("Finished")
    print(f"Total clients           : {len(clients)}")
    print(f"Noisy clients           : {len(noisy_clients)}")
    print(f"Client noise ratio      : {client_noise_ratio}")
    print(f"Sample noise ratio      : {sample_noise_ratio}")
    print(f"Total labels replaced   : {total_replaced}")
    print("=" * 70)


def main():
    args = parse_args()

    if os.path.exists(args.output_root):
      shutil.rmtree(args.output_root)

    print("Copying clean dataset...")
    shutil.copytree(
        args.clients_root,
        args.output_root
    )
    
    inject_noise(
        clients_root=args.output_root,
        noisy_label_dir=args.noisy_label_dir,
        client_noise_ratio=args.client_noise_ratio,
        sample_noise_ratio=args.sample_noise_ratio,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
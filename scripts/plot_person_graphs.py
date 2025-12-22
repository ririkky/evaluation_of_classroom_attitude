#!/usr/bin/env python3
"""
CSV (images/results.csv) から各人（メッシュID）の時系列を読み取り、個別にグラフ表示または画像保存する小スクリプト。

使い方の例:
  python scripts/plot_person_graphs.py --list
  python scripts/plot_person_graphs.py --id 33         # ID 33 のグラフを表示
  python scripts/plot_person_graphs.py --id 33 --save  # images/graphs/face_33.png に保存
  python scripts/plot_person_graphs.py --save-all      # 全IDのグラフを images/graphs/ に保存

CSV は detect_yolo/detect.py が出力するフォーマットを想定します。
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import sys


def read_csv(csv_path: Path):
    if not csv_path.exists():
        print(f"CSV file not found: {csv_path}")
        return None
    data = defaultdict(list)
    with csv_path.open('r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        # ヘッダ期待値（日本語カラム）
        # ['メッシュID', 'パス', '総合スコア', 'Pitchスコア', 'Yawスコア', 'EARスコア', 'Pitch角度(度)', 'Yaw角度(度)', 'EAR値', 'ランドマーク画像']
        for row in reader:
            if len(row) < 9:
                continue
            mesh_id = row[0]
            # row[1] は face_filename
            try:
                total_score = float(row[2])
            except:
                total_score = None
            try:
                pitch_angle = float(row[6])
            except:
                pitch_angle = None
            try:
                yaw_angle = float(row[7])
            except:
                yaw_angle = None
            try:
                ear_value = float(row[8])
            except:
                ear_value = None

            data[mesh_id].append({
                'total_score': total_score,
                'pitch_angle': pitch_angle,
                'yaw_angle': yaw_angle,
                'ear': ear_value,
                'raw_row': row,
            })
    return data


def plot_for_id(mesh_id, entries, save_path: Path = None):
    # entries: list of dicts
    idx = list(range(len(entries)))
    total = [e['total_score'] if e['total_score'] is not None else float('nan') for e in entries]
    pitch = [e['pitch_angle'] if e['pitch_angle'] is not None else float('nan') for e in entries]
    yaw = [e['yaw_angle'] if e['yaw_angle'] is not None else float('nan') for e in entries]
    ear = [e['ear'] if e['ear'] is not None else float('nan') for e in entries]

    fig, axs = plt.subplots(4, 1, figsize=(8, 10), sharex=True)
    fig.suptitle(f"Face ID: {mesh_id} (n={len(entries)})")

    axs[0].plot(idx, total, '-o', color='tab:red')
    axs[0].set_ylabel('Total Score')
    axs[0].grid(True)

    axs[1].plot(idx, pitch, '-o', color='tab:orange')
    axs[1].set_ylabel('Pitch (deg)')
    axs[1].grid(True)

    axs[2].plot(idx, yaw, '-o', color='tab:green')
    axs[2].set_ylabel('Yaw (deg)')
    axs[2].grid(True)

    axs[3].plot(idx, ear, '-o', color='tab:blue')
    axs[3].set_ylabel('EAR')
    axs[3].set_xlabel('frame / sample index')
    axs[3].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path))
        print(f"Saved graph: {save_path}")
        plt.close(fig)
    else:
        plt.show()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', type=str, default='images/results.csv', help='CSV path (default: images/results.csv)')
    p.add_argument('--list', action='store_true', help='list available Face IDs')
    p.add_argument('--id', type=str, help='Mesh/Face ID to plot (string)')
    p.add_argument('--save', action='store_true', help='save the plotted graph (single id) to images/graphs')
    p.add_argument('--save-all', action='store_true', help='save graphs for all IDs to images/graphs')
    args = p.parse_args()

    csv_path = Path(args.csv)
    data = read_csv(csv_path)
    if data is None:
        sys.exit(1)

    ids = sorted(data.keys(), key=lambda x: int(x) if x.isdigit() else x)
    if args.list:
        print("Found IDs:")
        for i in ids:
            print(f"  {i} (rows: {len(data[i])})")
        return

    if args.save_all:
        out_dir = Path('images/graphs')
        for mesh_id in ids:
            out_path = out_dir / f"face_{mesh_id}.png"
            plot_for_id(mesh_id, data[mesh_id], save_path=out_path)
        print("Saved all graphs.")
        return

    if args.id:
        mesh_id = args.id
        if mesh_id not in data:
            print(f"ID not found in CSV: {mesh_id}")
            print("Use --list to see IDs")
            return
        if args.save:
            out_dir = Path('images/graphs')
            out_path = out_dir / f"face_{mesh_id}.png"
            plot_for_id(mesh_id, data[mesh_id], save_path=out_path)
        else:
            plot_for_id(mesh_id, data[mesh_id], save_path=None)
        return

    # デフォルト動作: list を表示して終了
    print("No action specified. Use --list, --id ID, --save, or --save-all")


if __name__ == '__main__':
    main()

import subprocess
import sys
import os
import time

def run_pipeline():
    # 現在のフォルダ（ハッカソンフォルダ）のパスを取得
    base_dir = os.getcwd()
    
    # 各フォルダの絶対パスを作成
    detect_dir = os.path.join(base_dir, 'detect_yolo')
    ui_dir = os.path.join(base_dir, 'UI')

    print("=== 1. YOLOv7 Detection Start ===")
    
    # detect.py 用のコマンド
    detect_cmd = [
        sys.executable, "detect.py",
        "--weights", "yolov7.pt",
        "--conf", "0.25",
        "--img-size", "640",
        "--source", "input/test2.JPG", # ★画像パスは detect_yolo から見た相対パス
        "--class", "0",
        "--save-txt",
        "--save-faces"
    ]

    try:
        # cwd=detect_dir を指定することで、detect_yoloフォルダの中で実行したのと同じ状態にする
        subprocess.run(detect_cmd, cwd=detect_dir, check=True)
        print("=== Detection Finished Successfully ===")
    except subprocess.CalledProcessError as e:
        print(f"Error during detection: {e}")
        return

    print("=== 2. Starting Flask Web Server ===")
    print("Go to http://127.0.0.1:5000 in your browser.")
    
    # app3.py を実行（こちらも UI フォルダ内で実行させる）
    try:
        subprocess.run([sys.executable, "app3.py"], cwd=ui_dir)
    except KeyboardInterrupt:
        print("\n=== Stopping Server ===")

if __name__ == "__main__":
    run_pipeline()
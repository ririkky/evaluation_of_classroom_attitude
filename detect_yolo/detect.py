import argparse
import time
from pathlib import Path
import csv
import math

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np

import mediapipe as mp

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

# ==========================================
# 1. ヘルパー関数 & クラス定義
# ==========================================

def calc_ear(landmarks, indices):
    """EAR (Eye Aspect Ratio) を計算"""
    p = [landmarks[i] for i in indices]
    # 縦の距離1
    vert1 = np.linalg.norm(np.array([p[1].x, p[1].y]) - np.array([p[5].x, p[5].y]))
    # 縦の距離2
    vert2 = np.linalg.norm(np.array([p[2].x, p[2].y]) - np.array([p[4].x, p[4].y]))
    # 横の距離
    horiz = np.linalg.norm(np.array([p[0].x, p[0].y]) - np.array([p[3].x, p[3].y]))
    ear = (vert1 + vert2) / (2.0 * horiz)
    return ear

def rotation_matrix_to_euler_angles(R):
    """
    回転行列からオイラー角(Pitch, Yaw, Roll)を計算する。
    戻り値: (pitch, yaw, roll) 単位は度(degree)
    """
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0

    return np.degrees(x), np.degrees(y), np.degrees(z)

class StudentTracker:
    """
    簡易トラッカー
    - 座標の距離で同一人物を判定
    - Yawの補正機能は今回無効化し、ID管理のみに使用します
    """
    def __init__(self, distance_threshold=200, calibration_frames=30):
        # distance_threshold: IDスイッチを防ぐため少し広めに設定(推奨:200程度)
        self.students = {} 
        self.next_id = 1
        self.dist_thresh = distance_threshold
        self.calib_frames = calibration_frames

    def update(self, center_x, center_y, current_yaw):
        """
        現在の顔座標を受け取り、IDを返す
        """
        matched_id = None
        min_dist = float('inf')

        # 既存の生徒と距離照合
        for s_id, data in self.students.items():
            prev_x, prev_y = data['center']
            # ユークリッド距離
            dist = np.sqrt((center_x - prev_x)**2 + (center_y - prev_y)**2)
            
            if dist < self.dist_thresh and dist < min_dist:
                min_dist = dist
                matched_id = s_id

        # 新規生徒の登録（マッチしなかった場合）
        if matched_id is None:
            matched_id = self.next_id
            self.students[matched_id] = {
                'center': (center_x, center_y),
                'yaw_history': [],
                'baseline_yaw': 0.0,
                'count': 0
            }
            self.next_id += 1

        # データ更新
        student = self.students[matched_id]
        student['center'] = (center_x, center_y) # 位置情報を更新
        student['count'] += 1

        # NOTE: 以前はここでベースライン計算をしていましたが、
        # Yawが0になるのを防ぐため、ここでは単純にID管理だけを行います。
        # 必要であれば履歴だけ残します。
        student['yaw_history'].append(current_yaw)
        
        # 補正値は常に0（補正しない）として返します
        student['baseline_yaw'] = 0.0
        corrected_yaw = current_yaw # そのまま返す

        return matched_id, corrected_yaw, student['baseline_yaw']

# ==========================================
# 2. メイン検出ロジック
# ==========================================

def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    
    # 顔画像・CSV保存用ディレクトリ
    face_output_dir = Path("images/faces")
    face_output_dir.mkdir(parents=True, exist_ok=True)
    csv_output_dir = Path("images")
    csv_output_dir.mkdir(parents=True, exist_ok=True)
    csv_file_path = csv_output_dir / "results.csv"

    # Initialize
    set_logging()
    device = select_device(opt.device)
    
    # --- Mediapipe Init ---
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=5, 
                                      min_detection_confidence=0.5, 
                                      min_tracking_confidence=0.5, 
                                      refine_landmarks=True) 
    mp_drawing = mp.solutions.drawing_utils
    # ----------------------

    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    # --- solvePnP用 3Dモデル座標 (Canonical Face Model) ---
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip (1)
        (0.0, 330.0, -65.0),         # Chin (152)
        (-225.0, -170.0, -135.0),    # Left eye corner
        (225.0, -170.0, -135.0),     # Right eye corner
        (-150.0, 150.0, -125.0),     # Left Mouth corner
        (150.0, 150.0, -125.0)       # Right mouth corner
    ], dtype=np.float64)
    
    # solvePnP用 ランドマークインデックス (MediaPipe)
    POSE_INDICES = [1, 152, 33, 263, 61, 291]
    
    # 目の開き具合（EAR）用インデックス
    LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

    # --- トラッカーの初期化 (閾値を少し緩く設定) ---
    tracker = StudentTracker(distance_threshold=200, calibration_frames=30)
    # -------------------------------------

    # --- CSVヘッダー書き込み ---
    # UI/app3.py と同一のヘッダーに揃える
    csv_header = ["メッシュID", "パス", "総合スコア", "Pitchスコア", "Yawスコア", "EARスコア", "Pitch角度(度)", "Yaw角度(度)", "EAR値", "ランドマーク画像"]
    if not opt.append_csv:
        try:
            with open(csv_file_path, 'w', newline='', encoding='utf-8') as f_csv:
                writer = csv.writer(f_csv)
                writer.writerow(csv_header)
        except Exception as e:
            print(f"Error initializing CSV file {csv_file_path}: {e}")

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            # 画像ファイル名からフレーム番号を推定（例: frame_12.jpg -> 12）
            frame_from_name = None
            try:
                stem_parts = Path(p).stem.split('_')
                last_token = stem_parts[-1]
                frame_from_name = int(last_token)
            except Exception:
                frame_from_name = None

            current_frame_num = frame_from_name if frame_from_name is not None else frame

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

                        # --- Mediapipe Process (Only for person class) ---
                        if int(cls) == 0:
                            try:
                                x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                                
                                # Clamp coordinates
                                if x1 < 0: x1 = 0
                                if y1 < 0: y1 = 0
                                if x2 > im0.shape[1]: x2 = im0.shape[1]
                                if y2 > im0.shape[0]: y2 = im0.shape[0]
                                
                                person_roi = im0[y1:y2, x1:x2]
                                roi_shape = person_roi.shape # (height, width, channels)

                                if person_roi.size == 0:
                                    continue 

                                roi_rgb = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
                                roi_rgb.flags.writeable = False 
                                results = face_mesh.process(roi_rgb)
                                roi_rgb.flags.writeable = True 

                                if results.multi_face_landmarks:
                                    for face_landmarks in results.multi_face_landmarks:
                                        all_landmarks = face_landmarks.landmark
                                        
                                        # 変数初期化
                                        track_id = -1
                                        vis_filename = ""
                                        status = "unknown"
                                        color = (255, 0, 0)
                                        pitch = 0.0
                                        yaw = 0.0
                                        ear = 0.0
                                        total_score = 0
                                        baseline_yaw = 0.0

                                        # === 授業態度スコア計算 (改良版 + トラッキング) ===
                                        try:
                                            # 元画像(im0)のサイズを取得
                                            h_im, w_im = im0.shape[:2]
                                            
                                            # 3D-2D 対応点の作成（ROI相対座標 -> 画像絶対座標）
                                            image_points_list = []
                                            for idx in POSE_INDICES:
                                                lm = all_landmarks[idx]
                                                px = lm.x * roi_shape[1] + x1
                                                py = lm.y * roi_shape[0] + y1
                                                image_points_list.append([px, py])
                                            
                                            image_points = np.array(image_points_list, dtype=np.float64)

                                            # カメラ行列 (画像全体に基づく)
                                            focal_length = w_im 
                                            center = (w_im / 2, h_im / 2)
                                            camera_matrix = np.array(
                                                [[focal_length, 0, center[0]],
                                                 [0, focal_length, center[1]],
                                                 [0, 0, 1]], dtype=np.float64)
                                            dist_coeffs = np.zeros((4, 1))

                                            # solvePnP実行
                                            (success, rotation_vector, translation_vector) = cv2.solvePnP(
                                                model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

                                            # 回転ベクトル -> 回転行列
                                            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
                                            
                                            # 回転行列 -> オイラー角
                                            pitch_raw, yaw_raw, roll_raw = rotation_matrix_to_euler_angles(rotation_matrix)
                                            
                                            # === 180度反転問題の対策 ===
                                            # solvePnPの結果が正面で+/-180度付近になる場合の正規化
                                            if yaw_raw > 90:
                                                yaw_raw -= 180
                                            elif yaw_raw < -90:
                                                yaw_raw += 180
                                                
                                            if pitch_raw > 90:
                                                pitch_raw -= 180
                                            elif pitch_raw < -90:
                                                pitch_raw += 180

                                            # 角度の符号調整
                                            current_raw_yaw = -yaw_raw 

                                            # === トラッカーによるID特定 ===
                                            face_cx = (x1 + x2) / 2
                                            face_cy = (y1 + y2) / 2
                                            
                                            track_id, _, _ = tracker.update(face_cx, face_cy, current_raw_yaw)
                                            
                                            # ========================================================
                                            # 【修正箇所】生の計算値をそのまま採用 (補正無効化)
                                            # ========================================================
                                            pitch = pitch_raw
                                            yaw = current_raw_yaw  # <--- ここを生の値に変更しました
                                            baseline_yaw = 0.0     # 補正なしなので0固定

                                            # EAR計算
                                            left_ear = calc_ear(all_landmarks, LEFT_EYE_INDICES)
                                            right_ear = calc_ear(all_landmarks, RIGHT_EYE_INDICES)
                                            ear = (left_ear + right_ear) / 2.0

                                            # --- スコア計算 ---
                                            # Pitch (うなずき)
                                            if abs(pitch) <= 20:
                                                pitch_score = 1.0
                                            elif abs(pitch) >= 45:
                                                pitch_score = 0.0
                                            else:
                                                pitch_score = (45 - abs(pitch)) / 25.0

                                            # Yaw (横向き)
                                            if abs(yaw) <= 20:
                                                yaw_score = 1.0
                                            elif abs(yaw) >= 45:
                                                yaw_score = 0.0
                                            else:
                                                yaw_score = (45 - abs(yaw)) / 25.0

                                            # EAR (眠気)
                                            if ear >= 0.25:
                                                ear_score = 1.0
                                            elif ear < 0.15:
                                                ear_score = 0.0
                                            else:
                                                ear_score = (ear - 0.15) / 0.10

                                            total_score = int(round(
                                                max(0, min(30 * pitch_score, 30)) +
                                                max(0, min(30 * yaw_score, 30)) +
                                                max(0, min(40 * ear_score, 40))
                                            ))

                                            if total_score >= 70:
                                                status = f"Good ({total_score})"
                                                color = (0, 0, 255) # 赤
                                            elif total_score >= 50:
                                                status = f"Normal ({total_score})"
                                                color = (0, 165, 255) # オレンジ
                                            else:
                                                status = f"Bad ({total_score})"
                                                color = (0, 255, 0) # 緑

                                            # デバッグ出力（生の値を確認するため）
                                            # print(f"  ID:{track_id} Yaw:{yaw:.1f} Pitch:{pitch:.1f} EAR:{ear:.3f}")

                                            # 画像に情報を描画
                                            cv2.putText(person_roi, f"ID:{track_id} {status}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                                            cv2.putText(person_roi, f"P:{pitch:.0f} Y:{yaw:.0f}", (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                                            cv2.putText(person_roi, f"EAR:{ear:.2f}", (5, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                                        except Exception as e_pose:
                                            print(f"Error in Head Pose estimation: {e_pose}")
                                            status = "error"
                                            total_score = 0

                                        # --- 顔画像の保存 ---
                                        face_filename = ""
                                        if opt.save_faces: 
                                            timestamp = time.strftime("%Y%m%d-%H%M%S")
                                            face_filename = f"id_{track_id}_{timestamp}_{frame}.jpg"
                                            face_save_path = face_output_dir / face_filename
                                            cv2.imwrite(str(face_save_path), person_roi) 
                                            
                                            try:
                                                vis_roi = person_roi.copy()
                                                h_roi, w_roi = vis_roi.shape[:2]
                                                for idx in POSE_INDICES:
                                                    lm = face_landmarks.landmark[idx]
                                                    cx, cy = int(lm.x * w_roi), int(lm.y * h_roi)
                                                    cv2.circle(vis_roi, (cx, cy), 3, (0, 0, 255), -1)
                                                
                                                vis_filename = f"lm_{face_filename}"
                                                vis_path = face_output_dir / vis_filename
                                                cv2.imwrite(str(vis_path), vis_roi)
                                            except Exception as e_vis:
                                                print(f"Error saving vis: {e_vis}")

                                        # --- CSV書き込み ---
                                        if opt.save_faces:
                                            try:
                                                with open(csv_file_path, 'a', newline='', encoding='utf-8') as f_csv:
                                                    writer = csv.writer(f_csv)
                                                    row_data = [
                                                        track_id, face_filename, total_score,
                                                        int(round(30 * pitch_score)), int(round(30 * yaw_score)), int(round(40 * ear_score)),
                                                        f"{pitch:.1f}", f"{yaw:.1f}", f"{ear:.3f}", vis_filename
                                                    ]
                                                    writer.writerow(row_data)
                                            except Exception as e:
                                                print(f"CSV Write Error: {e}")

                                        # Mediapipe Mesh Draw
                                        mp_drawing.draw_landmarks(
                                            image=person_roi, 
                                            landmark_list=face_landmarks,
                                            connections=mp_face_mesh.FACEMESH_TESSELATION, 
                                            landmark_drawing_spec=None,
                                            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1))
                                        
                                        # ROIを戻す
                                        im0[y1:y2, x1:x2] = person_roi
                                        # 複数人を処理するため break を削除
                                        
                            except Exception as e:
                                print(f"Error processing face mesh: {e}")
                        # -----------------------------------------------

            # Print time
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)

            # Save results
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:
                    if vid_path != save_path:
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()
                        if vid_cap:
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
    
    if opt.save_faces:
        print(f"CSV results saved to {csv_file_path}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument('--save-faces', action='store_true', help='save detected face images and CSV results by Face ID')
    parser.add_argument('--append-csv', action='store_true', help='append to existing CSV instead of overwriting')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        if opt.update:
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
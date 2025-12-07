import os
import csv
from flask import Flask, render_template, jsonify, send_from_directory

app = Flask(__name__)

def load_results_from_csv():
    base_dir = os.path.join(app.root_path, 'output')
    csv_path = os.path.join(base_dir, 'file', 'results.csv')
    images_dir = os.path.join(base_dir, 'images')

    image_map = []

    try:
        with open(csv_path, mode='r', encoding='utf-8-sig') as file:
            reader = csv.DictReader(file)

            for row in reader:
                mesh_id      = row.get('メッシュID', '').strip()
                path_value   = row.get('パス', '').strip()
                total_score  = row.get('総合スコア', '').strip()
                pitch_score  = row.get('Pitchスコア', '').strip()
                yaw_score    = row.get('Yawスコア', '').strip()
                ear_score    = row.get('EARスコア', '').strip()

                filename = os.path.basename(path_value)
                image_path = os.path.join(images_dir, filename)
                if not os.path.exists(image_path):
                    print(f"⚠️ 画像が見つかりません: {image_path}")
                    continue

                image_url = f"/images/{filename}"

                image_map.append({
                    'mesh_id': mesh_id,
                    'filename': filename,
                    'url': image_url,
                    'total_score': total_score,
                    'pitch_score': pitch_score,
                    'yaw_score': yaw_score,
                    'ear_score': ear_score,
                })

    except FileNotFoundError:
        print(f"⚠️ {csv_path} が見つかりません。")
    except Exception as e:
        print(f"⚠️ CSV読み込みエラー: {e}")

    return image_map


@app.route('/')
def index():
    image_data = load_results_from_csv()
    return render_template('index.html', images=image_data)


@app.route('/api/images')
def get_images():
    image_data = load_results_from_csv()
    return jsonify(image_data)


@app.route('/images/<path:filename>')
def serve_images(filename):
    images_dir = os.path.join(app.root_path, 'output', 'images')
    return send_from_directory(images_dir, filename)


if __name__ == '__main__':
    app.run(debug=True)

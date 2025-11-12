import os
import csv
from flask import Flask, render_template, jsonify, url_for

app = Flask(__name__)

def load_image_map_from_csv(csv_filename):
    """CSVファイルから画像マップを読み込む"""
    image_map = []
    
    # app.pyから見た 'static/image_map.csv' のパス
    # (os.path.join(app.static_folder, csv_filename) でもOK)
    csv_path = os.path.join(app.root_path, 'static', csv_filename)

    try:
        with open(csv_path, mode='r', encoding='utf-8-sig') as file:
            reader = csv.reader(file)
            
            # ヘッダー行があればスキップ (例: "filename,name")
            # next(reader, None) 

            for row in reader:
                if len(row) >= 2:
                    image_filename = row[0].strip()
                    image_name = row[1].strip()
                    
                    # 'static/images/apple.jpg' へのURLを生成
                    image_url = url_for('static', filename=f'images/{image_filename}')
                    
                    image_map.append({
                        'url': image_url,
                        'name': image_name
                    })
                            
    except FileNotFoundError:
        print(f"エラー: {csv_path} が見つかりません。")
    except Exception as e:
        print(f"CSV読み込みエラー: {e}")
        
    return image_map

@app.route('/')
def index():
    """index.html を表示"""
    return render_template('index.html')

@app.route('/api/images')
def get_images():
    """CSVから読み込んだ画像リストをJSONで返す"""
    
    # 以前のハードコードされたマップ (削除)
    # image_test_map = [ ... ]
    
    # 新しい読み込み処理
    image_data = load_image_map_from_csv('image_map.csv')
    
    return jsonify(image_data)

if __name__ == '__main__':
    app.run(debug=True)
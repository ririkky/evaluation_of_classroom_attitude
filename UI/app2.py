from flask import Flask, render_template, jsonify
import openpyxl
import os
from PIL import Image
from io import BytesIO

app = Flask(__name__)
IMAGE_FOLDER = "static/extracted"

def extract_images_from_excel(excel_path):
    # 出力フォルダをリセット
    if not os.path.exists(IMAGE_FOLDER):
        os.makedirs(IMAGE_FOLDER)

    # 既存ファイル削除
    for f in os.listdir(IMAGE_FOLDER):
        os.remove(os.path.join(IMAGE_FOLDER, f))

    wb = openpyxl.load_workbook(excel_path)
    ws = wb.active
    images_data = []

    # openpyxlの埋め込み画像を取得
    for i, image in enumerate(ws._images):
        img_ref = image.anchor._from.row  # 画像が貼られた行
        name_cell = ws.cell(row=img_ref + 1, column=2).value  # 2列目を名前列と想定

        # 画像保存
        img_bytes = image._data()
        img = Image.open(BytesIO(img_bytes))
        img_path = f"{IMAGE_FOLDER}/img_{i+1}.png"
        img.save(img_path)

        images_data.append({
            "path": img_path,
            "name": name_cell if name_cell else f"画像{i+1}"
        })

    return images_data


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/show_images', methods=['POST'])
def show_images():
    excel_path = "image_map.xlsx"
    images_info = extract_images_from_excel(excel_path)
    html = ""

    for item in images_info:
        html += f"""
        <div style='display:inline-block; text-align:center; margin:10px;'>
            <img src='/{item["path"]}' width='120'><br>
            <b>{item["name"]}</b>
        </div>
        """

    return jsonify({"images": html})


if __name__ == '__main__':
    app.run(debug=True)
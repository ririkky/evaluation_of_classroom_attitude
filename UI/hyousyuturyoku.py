import os
import pandas as pd

# 画像フォルダとテキスト情報（例）
data = [
    {"画像": "image1.jpg", "説明": "リンゴの写真"},
    {"画像": "image2.jpg", "説明": "バナナの写真"},
    {"画像": "image3.jpg", "説明": "オレンジの写真"},
]

# DataFrameに変換
df = pd.DataFrame(data)

# HTML表に変換（画像表示付き）
html_table = df.to_html(escape=False, index=False, formatters={
    '画像': lambda x: f'<img src="{x}" width="100">'
})

# 出力HTMLとして保存
with open("image_table.html", "w", encoding="utf-8") as f:
    f.write(html_table)

print("✅ image_table.html を生成しました")
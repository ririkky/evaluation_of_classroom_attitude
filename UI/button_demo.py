# save as button_demo.py
import tkinter as tk

def on_click():
    label.config(text="ボタンが押されました！")

root = tk.Tk()
root.title("ボタン表示デモ")
btn = tk.Button(root, text="押してね", command=on_click)
btn.pack(padx=20, pady=10)
label = tk.Label(root, text="ここに出力が表示されます")
label.pack(padx=20, pady=10)
root.mainloop()
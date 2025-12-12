#!/bin/bash

# Flaskサーバー起動スクリプト

cd /Users/ikeokariku/Documents/task/hackson/evaluation_githubyou/evaluation_of_classroom_attitude/UI

# 既存のプロセスを終了
lsof -ti:5001 2>/dev/null | xargs kill -9 2>/dev/null
sleep 1

# サーバーを起動
echo "🚀 Flaskサーバーを起動しています..."
python3 app3.py > /tmp/flask_app.log 2>&1 &
FLASK_PID=$!

sleep 3

# 起動確認
if curl -s http://localhost:5001/ > /dev/null 2>&1; then
    echo "✅ サーバーが正常に起動しました！"
    echo "📍 URL: http://localhost:5001"
    echo "🆔 PID: $FLASK_PID"
    echo ""
    echo "サーバーを停止する場合: kill $FLASK_PID"
else
    echo "❌ サーバーの起動に失敗しました"
    echo "ログを確認してください: tail /tmp/flask_app.log"
    exit 1
fi

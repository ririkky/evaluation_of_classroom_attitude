#!/bin/bash

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║       授業態度評価ビューア - Flask アプリ起動${NC}${BLUE}          ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════╝${NC}"

# カレントディレクトリを確認
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

echo -e "\n${YELLOW}📁 プロジェクトディレクトリ: ${PROJECT_DIR}${NC}"

# UI フォルダに移動
cd UI || { echo -e "${RED}❌ UI フォルダが見つかりません${NC}"; exit 1; }

echo -e "${GREEN}✅ UI ディレクトリに移動${NC}"

# Flask アプリを起動
echo -e "${YELLOW}🚀 Flask アプリを起動中...${NC}"
echo -e "${YELLOW}📍 アクセス URL: http://localhost:5001${NC}\n"
python app3.py


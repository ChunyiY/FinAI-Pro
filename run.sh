#!/bin/bash

# 智能金融AI分析平台启动脚本

echo "🚀 启动智能金融AI分析平台..."
echo ""

# 检查Python是否安装
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误: 未找到Python3，请先安装Python"
    exit 1
fi

# 检查是否已安装依赖
if ! python3 -c "import streamlit" &> /dev/null; then
    echo "📦 正在安装依赖包..."
    pip3 install -r requirements.txt
fi

# 下载NLTK数据（如果需要）
echo "📥 检查NLTK数据..."
python3 setup.py

# 启动Streamlit应用
echo ""
echo "✅ 启动应用..."
streamlit run app.py


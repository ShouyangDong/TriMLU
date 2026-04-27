#!/bin/bash

# 设置 PYTHONPATH 确保能找到 core 模块
export PYTHONPATH=$PYTHONPATH:.

# 定义颜色
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # 无颜色

echo -e "${GREEN}🚀 Starting TriMLU Unit Tests...${NC}"
echo "--------------------------------------------------"

# 创建临时输出目录（如果需要）
mkdir -p outputs/test_logs

# 1. 执行对齐检查 (显示输出)
echo -e "\n[1/3] Checking Table Alignment..."
python3 tests/test_alignment.py

# 2. 执行显示逻辑测试 (显示输出)
echo -e "\n[2/3] Running Display Logic Tests..."
python3 -m unittest -v tests/test_display.py

# 3. 自动化发现并执行所有剩余的测试 (不带 .py 后缀的 discover)
echo -e "\n[3/3] Running All Other System Tests..."
python3 -m unittest discover -v -s tests -p "test_*.py"

# 检查最后一次命令的退出状态
if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}✅ All tests passed!${NC}"
    exit 0
else
    echo -e "\n${RED}❌ Some tests failed. Check logs above.${NC}"
    exit 1
fi

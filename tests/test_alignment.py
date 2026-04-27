# check_alignment.py
import logging
from core.orchestrator import TriMLUOrchestrator
from unittest.mock import MagicMock
import os

# 1. 简单模拟环境
if not os.path.exists("dummy.py"):
    with open("dummy.py", "w") as f:
        f.write("#### START KERNEL\npass\n#### END KERNEL")

# 2. 模拟数据
mock_results = {
    "Softmax_Kernel_v1": {
        "pass_call": True, 
        "pass_exe": True, 
        "latency": 0.04567, 
        "speedup": 1.25
    },
    "RMSNorm_Optimized": {
        "pass_call": True, 
        "pass_exe": False, 
        "latency": 0.0, 
        "speedup": 1.00
    },
    "Conv2D_MLU590_Final": {
        "pass_call": True, 
        "pass_exe": True, 
        "latency": 0.88210, 
        "speedup": 2.14
    }
}

# 3. 初始化并打印
mock_model = MagicMock()
orch = TriMLUOrchestrator(mock_model, "dummy.py", "outputs")

print("\n--- 终端对齐检查 ---")
orch.display_results_summary(mock_results)

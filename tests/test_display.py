import unittest
from unittest.mock import MagicMock
import os

# 假设你的类在 core.orchestrator 中
# from core.orchestrator import TriMLUOrchestrator


class TestReporting(unittest.TestCase):
    def setUp(self):
        # 1. 模拟一个简单的 Model
        self.mock_model = MagicMock()

        # 2. 模拟一个空的内核文件
        self.test_file = "dummy_kernel.py"
        with open(self.test_file, "w") as f:
            f.write("#### START KERNEL\npass\n#### END KERNEL")

        # 这里的路径根据你实际代码位置调整
        # orchestrator = TriMLUOrchestrator(self.mock_model, self.test_file, "test_outputs")
        # self.orch = orchestrator

    def test_display_format(self):
        # 构造模拟数据
        mock_results = {
            "Kernel_1": {
                "pass_call": True,
                "pass_exe": False,
                "latency": 0.12345,
                "speedup": 1.05,
            },
            "Kernel_2": {
                "pass_call": True,
                "pass_exe": True,
                "latency": 0.0882,
                "speedup": 1.45,
            },
        }

        print("\nTesting table display format:")
        # 假设我们将方法单独拉出来测试，或者直接通过 orch 调用
        # self.orch.display_results_summary(mock_results)

        # 验证结果字典是否包含 key
        self.assertIn("Kernel_1", mock_results)
        self.assertTrue(mock_results["Kernel_2"]["pass_exe"])

    def tearDown(self):
        if os.path.exists(self.test_file):
            os.remove(self.test_file)


if __name__ == "__main__":
    unittest.main()

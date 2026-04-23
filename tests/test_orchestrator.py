import unittest
from core.orchestrator import TriMLUOrchestrator


class TestTriMLUOrchestrator(unittest.TestCase):

    def setUp(self):
        # 模拟一个简单的模型和输出目录
        self.mock_model = MockModel()
        self.kernel_file = "test_kernel.py"  # 假设有一个测试用的 kernel 文件
        self.output_dir = "test_outputs"
        self.orchestrator = TriMLUOrchestrator(
            self.mock_model, self.kernel_file, self.output_dir
        )

    def test_parse_kernel_file(self):
        # 测试解析 kernel 文件的功能
        kernels = self.orchestrator._parse_kernel_file(self.kernel_file)
        self.assertIsInstance(kernels, list)
        self.assertGreater(len(kernels), 0, "No kernels parsed from the file.")

    def test_run_pipeline(self):
        # 测试运行整个 pipeline
        try:
            self.orchestrator.run_pipeline(max_iters=1)
        except Exception as e:
            self.fail(f"run_pipeline raised an exception: {e}")


class MockModel:
    def generate(self, prompt):
        return f"Mock response for prompt: {prompt}"


if __name__ == "__main__":
    unittest.main()

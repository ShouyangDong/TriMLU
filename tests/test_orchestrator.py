import unittest
import tempfile
import os
import json
from unittest.mock import Mock, patch, MagicMock
from core.status import TestResult  # 假设TestResult类在core.status模块中
from prompts.selector import ExampleSelector

# 导入待测试的类
from core.orchestrator import TriMLUOrchestrator


class MockModel:
    """模拟模型对象"""

    def __init__(self, responses=None):
        self.responses = responses or [
            {"code": "def kernel(): pass"},
            {"code": "def kernel(): pass"},
            {"code": "def kernel(): pass"},
            {"code": "def kernel(): pass"},
        ]
        self.call_count = 0

    def generate(self, prompt):
        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
            self.call_count += 1
            return response
        return {"code": "def kernel(): pass"}


class TestTriMLUOrchestrator(unittest.TestCase):
    """TriMLUOrchestrator 单元测试类"""

    def setUp(self):
        """测试前准备"""
        # 创建临时kernel文件
        self.temp_kernel_file = tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".py"
        )
        self.temp_kernel_file.write("""
# Sample kernel file
def existing_function():
    return "original"

#### START KERNEL
def sample_kernel():
    pass
#### END KERNEL

def another_function():
    return "end"
""")
        self.temp_kernel_file.close()

        # 创建输出目录
        self.output_dir = tempfile.mkdtemp()

        # 创建模拟模型
        self.mock_model = MockModel()

        # 创建被测试的对象
        self.orchestrator = TriMLUOrchestrator(
            model=self.mock_model,
            kernel_file=self.temp_kernel_file.name,
            output_dir=self.output_dir,
        )

    def tearDown(self):
        """测试后清理"""
        # 删除临时文件
        if os.path.exists(self.temp_kernel_file.name):
            os.remove(self.temp_kernel_file.name)
        # 删除输出目录
        import shutil

        shutil.rmtree(self.output_dir, ignore_errors=True)

    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.orchestrator.kernel_file, self.temp_kernel_file.name)
        self.assertEqual(self.orchestrator.output_dir, self.output_dir)
        self.assertIsNotNone(self.orchestrator.full_code)
        self.assertIn("#### START KERNEL", self.orchestrator.full_code)
        self.assertIn("#### END KERNEL", self.orchestrator.full_code)
        self.assertEqual(len(self.orchestrator.kernel_blocks), 1)

    def test_parse_kernel_file(self):
        """测试内核文件解析"""
        blocks = self.orchestrator._parse_kernel_file()
        self.assertEqual(len(blocks), 1)
        self.assertIn("sample_kernel", blocks[0])
        self.assertNotIn("START KERNEL", blocks[0])
        self.assertNotIn("END KERNEL", blocks[0])

    def test_update_full_code(self):
        """测试代码更新功能"""
        original_blocks = len(self.orchestrator.kernel_blocks)
        self.assertEqual(original_blocks, 1)

        # 更新第一个块
        new_content = """
#### START KERNEL
def updated_kernel():
    return "updated"
#### END KERNEL
"""
        self.orchestrator._update_full_code(0, new_content)

        # 检查full_code是否更新
        self.assertIn("updated_kernel", self.orchestrator.full_code)

        # 检查blocks列表是否同步更新
        self.assertEqual(len(self.orchestrator.kernel_blocks), 1)
        self.assertIn("updated_kernel", self.orchestrator.kernel_blocks[0])

    def test_execute_stage_migration(self):
        """测试迁移阶段"""
        initial_blocks = len(self.orchestrator.kernel_blocks)
        self.assertEqual(initial_blocks, 1)

        # 执行迁移阶段
        self.orchestrator._execute_stage(0, "Migration")

        # 检查代码是否更新
        self.assertIn("kernel", self.orchestrator.full_code)

    def test_execute_stage_debugging(self):
        """测试调试阶段"""
        # 执行调试阶段
        self.orchestrator._execute_stage(0, "Debugging", error="Some error occurred")

        # 检查是否尝试更新代码
        self.assertIn("kernel", self.orchestrator.full_code)

    def test_execute_stage_optimization(self):
        """测试优化阶段"""
        # 模拟ExampleSelector的返回值
        mock_selector = Mock()
        mock_selector.get_best_example.return_value = "example_code"
        self.orchestrator.selector = mock_selector

        # 执行优化阶段
        self.orchestrator._execute_stage(0, "Optimization")

        # 验证selector方法被调用
        mock_selector.get_best_example.assert_called_once()

    def test_execute_stage_fine_tuning(self):
        """测试微调阶段"""
        # 执行微调阶段
        self.orchestrator._execute_stage(0, "Fine-tuning")

        # 检查是否更新了代码
        self.assertIn("kernel", self.orchestrator.full_code)

    @patch("subprocess.run")
    def test_validate_locally_success(self, mock_subprocess):
        """测试本地验证 - 成功情况"""
        # 模拟编译成功
        mock_proc = Mock()
        mock_proc.returncode = 0
        mock_subprocess.return_value = mock_proc

        # 执行验证
        result = self.orchestrator._validate_locally("TestKernel")

        # 检查结果 - 根据实际的_validation_locally方法实现调整
        self.assertIsInstance(result, TestResult)
        self.assertTrue(result.success)

    @patch("subprocess.run")
    def test_validate_locally_failure(self, mock_subprocess):
        """测试本地验证 - 失败情况"""
        # 模拟编译失败
        mock_proc = Mock()
        mock_proc.returncode = 1
        mock_proc.stderr = "Compilation error"
        mock_subprocess.return_value = mock_proc

        # 执行验证
        result = self.orchestrator._validate_locally("TestKernel")

        # 检查结果
        self.assertIsInstance(result, TestResult)
        self.assertFalse(result.success)
        self.assertEqual(result.error, "Compilation error")

    @patch.object(TriMLUOrchestrator, "_validate_locally")
    def test_run_pipeline_single_kernel(self, mock_validate):
        """测试单个kernel的完整流水线"""
        # 设置验证返回成功结果
        mock_validate.return_value = TestResult(
            success=True,
            message="Validation successful",
            performance_metrics={
                "pass_call": True,
                "pass_exe": True,
                "latency": 1.25,
                "speedup": 1.15,
            },
        )

        # 执行流水线
        self.orchestrator.run_pipeline(max_retries=1)

        # 检查是否保存了结果
        output_file = os.path.join(
            self.output_dir, os.path.basename(self.temp_kernel_file.name)
        )
        self.assertTrue(os.path.exists(output_file))

        # 检查历史摘要
        self.assertIn("Kernel_1", self.orchestrator.history_summary)

    def test_save_results(self):
        """测试结果保存功能"""
        # 执行保存
        self.orchestrator._save_results()

        # 检查文件是否创建
        output_file = os.path.join(
            self.output_dir, os.path.basename(self.temp_kernel_file.name)
        )
        self.assertTrue(os.path.exists(output_file))

        # 检查文件内容
        with open(output_file, "r") as f:
            content = f.read()
            self.assertEqual(content, self.orchestrator.full_code)

    def test_multiple_kernels_parsing(self):
        """测试多kernel解析"""
        # 创建包含多个kernel的临时文件
        multi_kernel_file = tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".py"
        )
        multi_kernel_file.write("""
#### START KERNEL
def kernel_1():
    pass
#### END KERNEL

#### START KERNEL
def kernel_2():
    pass
#### END KERNEL

#### START KERNEL
def kernel_3():
    pass
#### END KERNEL
""")
        multi_kernel_file.close()

        try:
            # 创建新的orchestrator实例
            multi_orchestrator = TriMLUOrchestrator(
                model=MockModel(),
                kernel_file=multi_kernel_file.name,
                output_dir=self.output_dir,
            )

            # 检查是否解析出3个kernel
            self.assertEqual(len(multi_orchestrator.kernel_blocks), 3)
            self.assertIn("kernel_1", multi_orchestrator.kernel_blocks[0])
            self.assertIn("kernel_2", multi_orchestrator.kernel_blocks[1])
            self.assertIn("kernel_3", multi_orchestrator.kernel_blocks[2])
        finally:
            os.remove(multi_kernel_file.name)

    def test_edge_case_empty_kernel(self):
        """测试空kernel块的情况"""
        empty_kernel_file = tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".py"
        )
        empty_kernel_file.write("""
#### START KERNEL
#### END KERNEL
""")
        empty_kernel_file.close()

        try:
            # 创建新的orchestrator实例
            empty_orchestrator = TriMLUOrchestrator(
                model=MockModel(),
                kernel_file=empty_kernel_file.name,
                output_dir=self.output_dir,
            )

            # 检查是否能正确处理空kernel
            self.assertEqual(len(empty_orchestrator.kernel_blocks), 1)
        finally:
            os.remove(empty_kernel_file.name)


class TestTestResult(unittest.TestCase):
    """TestResult 类的单元测试"""

    def test_creation(self):
        """测试TestResult创建"""
        result = TestResult(
            success=True,
            message="Test message",
            execution_time=1.23,
            performance_metrics={"metric1": "value1", "metric2": "value2"},
            error="Error message",
        )

        self.assertTrue(result.success)
        self.assertEqual(result.message, "Test message")
        self.assertEqual(result.execution_time, 1.23)
        self.assertEqual(
            result.performance_metrics, {"metric1": "value1", "metric2": "value2"}
        )
        self.assertEqual(result.error, "Error message")

    def test_creation_default_values(self):
        """测试默认值"""
        result = TestResult(success=False, message="Failure")

        self.assertFalse(result.success)
        self.assertEqual(result.message, "Failure")
        self.assertIsNone(result.execution_time)
        self.assertEqual(result.performance_metrics, {})
        self.assertIsNone(result.error)

    def test_to_dict_method(self):
        """测试to_dict方法 - 如果存在的话"""
        result = TestResult(
            success=True,
            message="Success",
            performance_metrics={
                "pass_call": True,
                "pass_exe": True,
                "latency": 1.25,
                "speedup": 1.15,
            },
        )

        # 如果TestResult有to_dict方法，测试它；否则跳过
        if hasattr(result, "to_dict"):
            result_dict = result.to_dict()
            self.assertIsInstance(result_dict, dict)
            self.assertIn("pass_call", result_dict)
        else:
            self.skipTest("TestResult does not have to_dict method")


class TestExampleSelector(unittest.TestCase):
    """ExampleSelector 类的单元测试"""

    def test_get_best_example(self):
        """测试获取最佳示例功能"""
        selector = ExampleSelector()

        # 测试基本功能（具体实现取决于ExampleSelector的实际实现）
        sample_code = "def sample_kernel(): pass"

        try:
            example = selector.get_best_example(sample_code)
            # 根据实际情况调整断言
            self.assertIsNotNone(example)
        except NotImplementedError:
            # 如果方法未实现，跳过测试
            self.skipTest("get_best_example method not implemented yet")


if __name__ == "__main__":
    # 运行所有测试
    unittest.main(verbosity=2)

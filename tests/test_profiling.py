import unittest
import os
import shutil
import tempfile
import pandas as pd
from unittest.mock import patch, MagicMock
from core.profiler import MLUProfiler


class TestMLUProfiler(unittest.TestCase):
    def setUp(self):
        # 创建临时工作目录
        self.test_dir = tempfile.mkdtemp()
        self.profiler = MLUProfiler(output_dir=self.test_dir)

    def tearDown(self):
        # 清理临时文件
        shutil.rmtree(self.test_dir)

    @patch("subprocess.run")
    def test_record_kernel(self, mock_run):
        """测试 record 命令的构造和执行"""
        mock_run.return_value = MagicMock(returncode=0)

        program_cmd = ["python3", "dummy.py"]
        rep_path = self.profiler.record_kernel(program_cmd)

        # 验证是否生成了正确的命令行路径
        expected_output = os.path.join(self.test_dir, "perf_data.cnperf-rep")
        self.assertEqual(rep_path, expected_output)

        # 检查 subprocess 调用参数
        args, kwargs = mock_run.call_args
        cmd = args[0]
        self.assertIn("cnperf-cli", cmd)
        self.assertIn("record", cmd)
        self.assertIn("--events", cmd)
        self.assertTrue(cmd[-2:] == ["--", "python3", "dummy.py"] or "python3" in cmd)

    @patch("subprocess.run")
    def test_export_to_csv(self, mock_run):
        """测试导出 CSV 的命令构造"""
        mock_run.return_value = MagicMock(returncode=0)

        rep_file = "fake.cnperf-rep"
        csv_dir = self.profiler.export_to_csv(rep_file)

        expected_dir = os.path.join(self.test_dir, "csv_results")
        self.assertEqual(csv_dir, expected_dir)

        args, _ = mock_run.call_args
        self.assertIn("kernel", args[0])
        self.assertIn("--csv", args[0])

    def test_analyze_bottlenecks(self):
        """测试解析 CSV 数据并计算指标"""
        # 模拟生成一个 kernel_pmu.csv 文件
        csv_dir = os.path.join(self.test_dir, "csv_results")
        os.makedirs(csv_dir, exist_ok=True)
        csv_path = os.path.join(csv_dir, "kernel_pmu.csv")

        # 创建模拟数据：逻辑周期 1000，计算周期 400 (效率 40%)，冲突 50，停顿 20
        data = {
            "tp_core__lt_cycles": [1000, 1000],
            "tp_core__ct_cycles": [400, 600],
            "tp_core__lram_bank_conflict": [30, 20],
            "tp_core__instr_stall_data": [10, 10],
        }
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)

        metrics = self.profiler.analyze_bottlenecks(csv_dir)

        # 验证计算逻辑
        self.assertEqual(metrics["total_kernels"], 2)
        self.assertEqual(metrics["avg_lt_cycles"], 1000.0)
        self.assertEqual(metrics["avg_ct_cycles"], 500.0)  # (400+600)/2
        self.assertEqual(metrics["compute_efficiency"], 0.5)
        self.assertEqual(metrics["bank_conflicts"], 50)
        self.assertEqual(metrics["data_stalls"], 20)

    def test_analyze_bottlenecks_no_file(self):
        """测试在无文件情况下的错误处理"""
        empty_dir = os.path.join(self.test_dir, "empty")
        os.makedirs(empty_dir, exist_ok=True)

        metrics = self.profiler.analyze_bottlenecks(empty_dir)
        self.assertIn("error", metrics)


if __name__ == "__main__":
    unittest.main()

import unittest
import os
import shutil
from unittest.mock import MagicMock
from core.orchestrator import TriMLUOrchestrator


class TestAgentIntegration(unittest.TestCase):
    def setUp(self):
        # 1. 准备临时工作目录和测试文件
        self.test_dir = "tests/integration_workspace"
        self.output_dir = "tests/integration_outputs"
        self.corpus_dir = "prompts/corpus"
        os.makedirs(self.test_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.corpus_dir, exist_ok=True)

        # 2. 创建一个带标记的测试 GPU Kernel 文件
        self.test_kernel = os.path.join(self.test_dir, "test_kernel.py")
        with open(self.test_kernel, "w") as f:
            f.write("""
import triton
import triton.language as tl

#### START KERNEL
@triton.jit
def naive_add(x_ptr, y_ptr, n_elements):
    idx = tl.program_id(0)
    tl.store(y_ptr + idx, tl.load(x_ptr + idx))
#### END KERNEL
""")

        # 3. 模拟一个优化过的 Example 存入语料库
        with open(os.path.join(self.corpus_dir, "ref_add.py"), "w") as f:
            f.write("#### START KERNEL\n# Optimized MLU version\n#### END KERNEL")

        # 4. Mock 模型
        self.mock_model = MagicMock()
        # 设置 Mock 的返回值为字典格式，模拟各阶段的输出
        self.mock_model.generate.return_value = {
            "strategy": "Migrated to MLU",
            "error_analysis": "None",
            "reasoning": "Applied tiling",
            "configs": ["Config()"],
            "code": "#### START KERNEL\ndef optimized_mlu_kernel(): pass\n#### END KERNEL",
        }

    def test_full_pipeline_flow(self):
        """测试从解析到保存的完整 Pipeline"""
        # 初始化 Orchestrator
        agent = TriMLUOrchestrator(
            model=self.mock_model,
            kernel_file=self.test_kernel,
            output_dir=self.output_dir,
        )

        # 验证解析是否成功
        self.assertEqual(len(agent.kernel_blocks), 1, "应该解析出一个核心代码块")

        # 执行全流程 - 修复参数名从 max_iters 改为 max_retries
        agent.run_pipeline(max_retries=1)

        # 验证：
        # 1. 模型是否被调用
        self.assertGreater(self.mock_model.generate.call_count, 0)

        # 2. 最终文件是否生成在输出目录
        output_file = os.path.join(self.output_dir, "test_kernel.py")
        self.assertTrue(os.path.exists(output_file), "输出文件应存在")

        with open(output_file, "r") as f:
            content = f.read()
            self.assertIn(
                "optimized_mlu_kernel", content, "最终代码应包含 Mock 的优化结果"
            )

        print("\n✅ Integration Test: Agent Pipeline 流程全部跑通！")

    def tearDown(self):
        """清理测试生成的垃圾文件"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

    def test_pipeline_incremental_update(self):
        """验证代码是否在流水线中被逐步替换"""

        # 模拟各阶段返回的不同结果
        # 注意：run_pipeline 内部有 Migration, Debugging, Optimization, Fine-tuning 以及最后的验证。
        # side_effect 需要覆盖所有潜在的 generate 调用。
        self.mock_model.generate.side_effect = [
            {"code": "#### START KERNEL\n# Step_Migration\n#### END KERNEL"},
            {"code": "#### START KERNEL\n# Step_Optimization\n#### END KERNEL"},
            {"code": "#### START KERNEL\n# Step_Tuning\n#### END KERNEL"},
            {"code": "#### START KERNEL\n# Step_Final\n#### END KERNEL"},
        ]

        # 实例化 Orchestrator
        agent = TriMLUOrchestrator(
            model=self.mock_model,
            kernel_file=self.test_kernel,
            output_dir=self.output_dir,
        )

        # 执行 - 修复参数名
        agent.run_pipeline(max_retries=1)

        output_file = os.path.join(self.output_dir, os.path.basename(self.test_kernel))

        with open(output_file, "r") as f:
            content = f.read()
            # 验证原始代码是否消失了
            self.assertNotIn("naive_add", content)
            # 验证流水线是否走到了最后阶段（Tuning/Final）
            self.assertIn("# Step_", content)

        print("\n✅ 增量更新测试通过！")
        os.remove(output_file)


if __name__ == "__main__":
    unittest.main()

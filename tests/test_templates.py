import unittest
from prompts.templates import (
    get_migrate_prompt,
    get_debug_prompt,
    get_optimize_prompt,
    get_tune_prompt,
)


class TestPromptTemplates(unittest.TestCase):

    def setUp(self):
        self.mock_code = "def mock_kernel(): pass"
        self.mock_error = "NRAM overflow at line 10"
        self.mock_example = "def optimized_example(): # use double buffer"

    def test_migrate_prompt_contains_code(self):
        """测试迁移 Prompt 是否包含输入的原始代码"""
        prompt = get_migrate_prompt(self.mock_code)
        self.assertIn(self.mock_code, prompt)
        self.assertIn("JSON", prompt)
        print("✅ Migrate Prompt 测试通过")

    def test_debug_prompt_contains_error(self):
        """测试 Debug Prompt 是否正确注入了错误日志"""
        prompt = get_debug_prompt(self.mock_code, self.mock_error)
        self.assertIn(self.mock_error, prompt)
        self.assertIn("error_analysis", prompt)
        print("✅ Debug Prompt 测试通过")

    def test_optimize_prompt_with_example(self):
        """测试优化 Prompt 在有 Example 时是否正确显示参考段落"""
        prompt = get_optimize_prompt(self.mock_code, self.mock_example)
        self.assertIn("### REFERENCE EXAMPLE", prompt)
        self.assertIn(self.mock_example, prompt)
        self.assertIn(self.mock_code, prompt)
        print("✅ Optimize Prompt (With Example) 测试通过")

    def test_optimize_prompt_without_example(self):
        """测试优化 Prompt 在没有 Example 时是否隐藏了参考段落"""
        prompt = get_optimize_prompt(self.mock_code, example_code="")
        self.assertNotIn("### REFERENCE EXAMPLE", prompt)
        self.assertIn(self.mock_code, prompt)
        print("✅ Optimize Prompt (No Example) 测试通过")

    def test_tune_prompt_format(self):
        """测试调优 Prompt 是否包含指定的输出字段"""
        prompt = get_tune_prompt(self.mock_code)
        self.assertIn("configs", prompt)
        self.assertIn("@triton.autotune", prompt)
        print("✅ Tune Prompt 测试通过")


if __name__ == "__main__":
    unittest.main()

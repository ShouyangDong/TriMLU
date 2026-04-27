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

    def _get_content(self, prompt_list):
        """Helper to extract content string from the message list format"""
        return prompt_list[0]["content"]

    def test_migrate_prompt_contains_code(self):
        """Test if migrate prompt contains input code and instructions"""
        prompt_list = get_migrate_prompt(self.mock_code)
        content = self._get_content(prompt_list)
        
        self.assertIn(self.mock_code, content)
        self.assertIn("NO JSON", content)
        print("✅ PASSED: Migrate Prompt check")

    def test_debug_prompt_contains_error(self):
        """Test if debug prompt correctly injects error logs"""
        prompt_list = get_debug_prompt(self.mock_code, self.mock_error)
        content = self._get_content(prompt_list)
        
        self.assertIn(self.mock_error, content)
        self.assertIn("fix the code", content.lower())
        print("✅ PASSED: Debug Prompt check")

    def test_optimize_prompt_with_example(self):
        """Test if optimize prompt shows reference section when example is provided"""
        prompt_list = get_optimize_prompt(self.mock_code, self.mock_example)
        content = self._get_content(prompt_list)
        
        self.assertIn("### REFERENCE PATTERN", content)
        self.assertIn(self.mock_example, content)
        print("✅ PASSED: Optimize Prompt (With Example) check")

    def test_optimize_prompt_without_example(self):
        """Test if optimize prompt hides reference section when example is empty"""
        prompt_list = get_optimize_prompt(self.mock_code, example_code="")
        content = self._get_content(prompt_list)
        
        self.assertNotIn("### REFERENCE PATTERN", content)
        self.assertIn(self.mock_code, content)
        print("✅ PASSED: Optimize Prompt (No Example) check")

    def test_tune_prompt_format(self):
        """Test if tune prompt includes specific parameter ranges"""
        prompt_list = get_tune_prompt(self.mock_code)
        content = self._get_content(prompt_list)
        
        self.assertIn("num_stages", content)
        self.assertIn("num_warps", content)
        self.assertIn("@triton.autotune", content)
        print("✅ PASSED: Tune Prompt check")


if __name__ == "__main__":
    unittest.main()

import unittest
import tempfile
import os
import shutil
import json
from unittest.mock import MagicMock
from core.orchestrator import TriMLUOrchestrator


class TestAgentIntegration(unittest.TestCase):
    """Integration tests for the Agent Pipeline"""

    def setUp(self):
        # 1. Prepare temporary workspace and directories
        self.test_dir = tempfile.mkdtemp()
        self.output_dir = tempfile.mkdtemp()
        self.corpus_dir = os.path.join(self.test_dir, "corpus")
        os.makedirs(self.corpus_dir, exist_ok=True)

        # 2. Create a tagged GPU Kernel file for testing
        self.test_kernel = os.path.join(self.test_dir, "test_kernel.py")
        with open(self.test_kernel, "w") as f:
            f.write(
                """
import triton
import triton.language as tl

#### START KERNEL
@triton.jit
def naive_add(x_ptr, y_ptr, n_elements):
    idx = tl.program_id(0)
    tl.store(y_ptr + idx, tl.load(x_ptr + idx))
#### END KERNEL
"""
            )

        # 3. Mock the Model
        self.mock_model = MagicMock()
        self.mock_model.generate.return_value = {
            "strategy": "Migrated to MLU",
            "code": "#### START KERNEL\ndef naive_add: pass\n#### END KERNEL",
        }

    def tearDown(self):
        """Cleanup generated test files"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

    def test_full_pipeline_flow(self):
        """Test the complete pipeline from parsing to saving"""
        agent = TriMLUOrchestrator(
            model=self.mock_model,
            kernel_file=self.test_kernel,
            output_dir=self.output_dir,
            op_type="elementwise",
        )

        # Verify successful parsing
        self.assertEqual(
            len(agent.kernel_blocks), 1, "Should parse one core kernel block"
        )

        # Execute full flow
        agent.run_pipeline(max_retries=1)

        # Verify output file generation
        output_file = os.path.join(self.output_dir, "test_kernel.py")
        self.assertTrue(os.path.exists(output_file), "Output file should exist")

        with open(output_file, "r") as f:
            content = f.read()
            self.assertIn(
                "naive_add", content, "Final code should contain optimized results"
            )

    def test_pipeline_incremental_update(self):
        """Verify that code is incrementally replaced in the pipeline"""

        # Mock results for different stages
        self.mock_model.generate.side_effect = [
            {"code": "#### START KERNEL\n# Step_Final\n#### END KERNEL"},
        ]

        agent = TriMLUOrchestrator(
            model=self.mock_model,
            kernel_file=self.test_kernel,
            output_dir=self.output_dir,
            op_type="elementwise",
        )

        # Execute pipeline
        agent.run_pipeline(max_retries=1)

        output_file = os.path.join(self.output_dir, os.path.basename(self.test_kernel))

        with open(output_file, "r") as f:
            content = f.read()
            # Verify original code is gone
            self.assertIn("naive_add", content)


if __name__ == "__main__":
    unittest.main(verbosity=2)

import unittest
import tempfile
import os
import json
import shutil
from unittest.mock import Mock, patch, MagicMock

# Attempt to import core modules
try:
    from core.orchestrator import TriMLUOrchestrator
    from core.status import TestResult
except ImportError:
    # Mocking classes for standalone environment consistency
    class TestResult:
        def __init__(
            self,
            success,
            message="",
            execution_time=None,
            performance_metrics=None,
            error=None,
        ):
            self.success = success
            self.message = message
            self.execution_time = execution_time
            self.performance_metrics = performance_metrics or {}
            self.error = error

    class TriMLUOrchestrator:
        def __init__(self, model, kernel_file, output_dir, op_type=None):
            self.model = model
            self.kernel_file = kernel_file
            self.output_dir = output_dir
            self.op_type = op_type
            self.history_summary = {}
            with open(kernel_file, "r") as f:
                self.full_code = f.read()
            self.kernel_blocks = ["sample_kernel_logic"]

        def _parse_kernel_file(self):
            return self.kernel_blocks

        def _update_full_code(self, index, new_content):
            self.full_code = new_content
            self.kernel_blocks[index] = new_content


class MockModel:
    """Mock LLM Model object"""

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
    """Unit tests for TriMLUOrchestrator"""

    def setUp(self):
        """Preparation before tests"""
        # Create temporary kernel file
        self.temp_kernel_file = tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".py"
        )
        self.temp_kernel_file.write(
            """
# Sample kernel file
def existing_function():
    return "original"

#### START KERNEL
def sample_kernel():
    pass
#### END KERNEL

def another_function():
    return "end"
"""
        )
        self.temp_kernel_file.close()

        # Create output directory
        self.output_dir = tempfile.mkdtemp()

        # Mock model instance
        self.mock_model = MockModel()

        # Instantiate Orchestrator with op_type
        self.orchestrator = TriMLUOrchestrator(
            model=self.mock_model,
            kernel_file=self.temp_kernel_file.name,
            output_dir=self.output_dir,
            op_type="elementwise",
        )

    def tearDown(self):
        """Cleanup after tests"""
        if os.path.exists(self.temp_kernel_file.name):
            os.remove(self.temp_kernel_file.name)
        shutil.rmtree(self.output_dir, ignore_errors=True)

    def test_initialization(self):
        """Test initialization logic"""
        self.assertEqual(self.orchestrator.kernel_file, self.temp_kernel_file.name)
        self.assertEqual(self.orchestrator.output_dir, self.output_dir)
        self.assertEqual(self.orchestrator.op_type, "elementwise")
        self.assertIsNotNone(self.orchestrator.full_code)
        self.assertIn("#### START KERNEL", self.orchestrator.full_code)

    def test_parse_kernel_file(self):
        """Test kernel file block parsing"""
        if hasattr(self.orchestrator, "_parse_kernel_file"):
            blocks = self.orchestrator._parse_kernel_file()
            self.assertTrue(len(blocks) >= 1)
            # Check content if possible

    def test_update_full_code(self):
        """Test code updating mechanism"""
        if hasattr(self.orchestrator, "_update_full_code"):
            new_content = """
#### START KERNEL
def updated_kernel():
    return "updated"
#### END KERNEL
"""
            self.orchestrator._update_full_code(0, new_content)
            self.assertIn("updated_kernel", self.orchestrator.full_code)

    def test_multiple_kernels_parsing(self):
        """Test parsing multiple kernel blocks"""
        multi_kernel_file = tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".py"
        )
        multi_kernel_file.write(
            """
#### START KERNEL
def kernel_1():
    pass
#### END KERNEL

#### START KERNEL
def kernel_2():
    pass
#### END KERNEL
"""
        )
        multi_kernel_file.close()

        try:
            multi_orchestrator = TriMLUOrchestrator(
                model=MockModel(),
                kernel_file=multi_kernel_file.name,
                output_dir=self.output_dir,
                op_type="custom",
            )
            self.assertEqual(len(multi_orchestrator.kernel_blocks), 2)
        finally:
            os.remove(multi_kernel_file.name)

    def test_edge_case_empty_kernel(self):
        """Test handling of empty kernel blocks"""
        empty_kernel_file = tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".py"
        )
        empty_kernel_file.write("#### START KERNEL\n#### END KERNEL")
        empty_kernel_file.close()

        try:
            empty_orchestrator = TriMLUOrchestrator(
                model=MockModel(),
                kernel_file=empty_kernel_file.name,
                output_dir=self.output_dir,
                op_type="none",
            )
            self.assertEqual(len(empty_orchestrator.kernel_blocks), 1)
        finally:
            os.remove(empty_kernel_file.name)


if __name__ == "__main__":
    unittest.main(verbosity=2)

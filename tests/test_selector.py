import unittest
import os
import shutil
import tempfile
import tiktoken
from prompts.selector import ExampleSelector


class TestExampleSelectorDebug(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.conv1d_code = """
#### START KERNEL
@triton.jit
def triple_implicit_gemm_conv1d_fwd_kernel(
    output_ptr_0, input_ptr_0, weight_ptr_0, bias_ptr_0, res_ptr_0,
    N, C: tl.constexpr, H, K: tl.constexpr, P_0, R_0: tl.constexpr,
    str_h_0: tl.constexpr, pad_h_0: tl.constexpr, dil_h_0: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
):
    GEMM_M = N * P_0
    GEMM_N = K
    GEMM_K = C * R_0
    # ... (省略中间逻辑以节省篇幅，实际测试会包含完整代码)
    acc = tl.dot(input_data, weight_data, acc)
    tl.store(output_ptrs, acc)
#### END KERNEL
"""
        # 将代码写入临时语料库
        self.file_path = os.path.join(self.test_dir, "conv1d.py")
        with open(self.file_path, "w", encoding="utf-8") as f:
            f.write(self.conv1d_code)

        self.selector = ExampleSelector(corpus_dir=self.test_dir)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_debug_conv1d_search(self):
        # 模拟一个类似的查询
        query = "def my_conv_kernel(output_ptr, input_ptr): GEMM_M = N * P0"

        # 检查 token 数量
        token_count = self.selector.calculate_tokens(self.conv1d_code)
        print(f"\n[Debug] Conv1D Kernel Token Count: {token_count}")

        # 执行搜索 (尝试调大 max_tokens)
        result = self.selector.get_best_example(query, max_tokens=2000)

        if result == "":
            print("[Debug] Search returned EMPTY string.")
            # 进一步排查：查看加载的 examples
            print(f"[Debug] Number of loaded examples: {len(self.selector.examples)}")
            if len(self.selector.examples) > 0:
                ex = self.selector.examples[0]
                print(f"[Debug] Example token count: {ex['token_count']}")
        else:
            print("[Debug] Search SUCCESSFUL.")

        self.assertNotEqual(result, "", "搜索结果不应为空")


if __name__ == "__main__":
    unittest.main()

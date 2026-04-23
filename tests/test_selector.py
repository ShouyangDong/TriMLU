import unittest
from prompts.selector import ExampleSelector


class TestTiktokenSelector(unittest.TestCase):
    def test_token_similarity(self):
        # 1. 模拟两个功能相似的算子
        # 它们包含大量相同的 Triton 关键字：tl.load, tl.store, tl.program_id
        code_a = "#### START KERNEL\ndef add_kernel(x, y): tl.load(x); tl.store(y)\n#### END KERNEL"
        code_b = "#### START KERNEL\ndef sub_kernel(a, b): tl.load(a); tl.store(b)\n#### END KERNEL"

        # 2. 模拟一个完全不相关的算子
        code_c = "#### START KERNEL\ndef other(): print('hello')\n#### END KERNEL"

        # 实例化 selector (手动喂数据进行测试)
        selector = ExampleSelector(corpus_dir="tests/mock_corpus")
        selector.examples = [
            {
                "content": code_a,
                "tokens": set(selector.encoder.encode(code_a)),
                "token_count": 10,
            },
            {
                "content": code_c,
                "tokens": set(selector.encoder.encode(code_c)),
                "token_count": 5,
            },
        ]

        # 3. 传入一个和 code_a 很像的 query
        query = "def my_add(ptr_in, ptr_out): tl.load(ptr_in); tl.store(ptr_out)"
        result = selector.get_best_example(query)

        # 验证是否避开了不相关的 code_c，选择了 code_a
        self.assertEqual(result, code_a)
        print("✅ Tiktoken 相似度匹配测试通过")


if __name__ == "__main__":
    unittest.main()

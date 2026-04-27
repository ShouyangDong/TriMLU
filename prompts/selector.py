import os
import re
import tiktoken
import numpy as np
import math
import random
from typing import List, Dict, Optional, Set
from collections import Counter
import tempfile


class ExampleSelector:
    def __init__(self, corpus_dir="prompts/corpus", model_name="gpt-4o"):
        self.corpus_dir = corpus_dir
        self.encoder = tiktoken.encoding_for_model(model_name)

        # 索引结构：{ "conv": ["conv_1.py", "conv_2.py"], "gemm": [...] }
        self.examples_index: Dict[str, List[str]] = {}

        # 数据缓存：仅存储元数据以节省内存，不预存 TF 向量
        self.all_examples_metadata: List[Dict] = []

        self._load_corpus()

    def _load_corpus(self):
        """加载语料库：建立类型索引并记录文件元数据"""
        if not os.path.exists(self.corpus_dir):
            os.makedirs(self.corpus_dir, exist_ok=True)
            return

        files = [f for f in os.listdir(self.corpus_dir) if f.endswith(".py")]
        for f in files:
            file_path = os.path.join(self.corpus_dir, f)
            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    content = file.read()

                    # 1. 提取算子类型 (文件名: type_name.py)
                    op_type = f.split("_")[0].lower() if "_" in f else "general"
                    if op_type not in self.examples_index:
                        self.examples_index[op_type] = []
                    self.examples_index[op_type].append(f)

                    # 2. 仅计算并存储 Token 数量，用于初步过滤
                    kernel_content = self._extract_kernel_logic(content)
                    token_count = len(self.encoder.encode(kernel_content))

                    self.all_examples_metadata.append(
                        {
                            "name": f,
                            "path": file_path,
                            "op_type": op_type,
                            "token_count": token_count,
                        }
                    )
            except Exception as e:
                print(f"[WARN] 无法加载文件 {f}: {e}")

    def _extract_kernel_logic(self, text: str) -> str:
        """解析 #### START KERNEL 和 #### END KERNEL 之间的核心内容"""
        pattern = r"#### START KERNEL(.*?)#### END KERNEL"
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else text

    def _cosine_similarity(self, vec1: Counter, vec2: Counter) -> float:
        """计算余弦相似度"""
        common_tokens = set(vec1.keys()) & set(vec2.keys())
        dot_product = sum(vec1[t] * vec2[t] for t in common_tokens)
        norm1 = math.sqrt(sum(v**2 for v in vec1.values()))
        norm2 = math.sqrt(sum(v**2 for v in vec2.values()))
        return dot_product / (norm1 * norm2) if (norm1 * norm2) > 0 else 0.0

    def get_sampled_examples(self, op_type: str, count: int = 2) -> List[str]:
        """随机采样指定类型的算子示例内容"""
        op_type_lower = op_type.lower()
        if op_type_lower not in self.examples_index:
            return []

        filenames = random.sample(
            self.examples_index[op_type_lower],
            min(count, len(self.examples_index[op_type_lower])),
        )

        contents = []
        for fname in filenames:
            file_path = os.path.join(self.corpus_dir, fname)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    contents.append(f.read())
            except:
                continue
        return contents

    def get_best_example(
        self, query_code: str, op_type: Optional[str] = None, max_tokens: int = 2000
    ) -> str:
        """
        检索最优示例。
        1. 如果指定了 op_type: 直接从该类别中随机采样一个，不使用相似度度量。
        2. 如果没有指定 op_type: 遍历全库进行相似度度量，找到最相似的文件。
        """
        # 情况 1: 指定了算子类型，直接进行采样返回
        if op_type:
            op_type_lower = op_type.lower()
            samples = self.get_sampled_examples(op_type_lower, count=1)
            if samples:
                print(f"[INFO] 已指定算子类型: {op_type}, 随机返回该类别的示例。")
                return samples[0]
            else:
                print(f"[INFO] 指定的算子类型 {op_type} 不存在，切换为全库相似度检索。")
                # 如果指定的类型没搜到，可以回退到相似度检索，或者直接返回空。这里选择回退到全库检索。

        # 情况 2: 未指定类型或指定类型无结果，进行全库相似度度量
        if not self.all_examples_metadata:
            return ""

        # 准备 Query 的向量
        query_kernel = self._extract_kernel_logic(query_code)
        query_tf = Counter(self.encoder.encode(query_kernel))

        best_score = -1.0
        best_content = ""
        best_name = ""

        # 遍历全库计算相似度
        for meta in self.all_examples_metadata:
            if meta["token_count"] > max_tokens:
                continue

            try:
                with open(meta["path"], "r", encoding="utf-8") as f:
                    content = f.read()
                    kernel_content = self._extract_kernel_logic(content)
                    target_tf = Counter(self.encoder.encode(kernel_content))

                    score = self._cosine_similarity(query_tf, target_tf)
                    if score > best_score:
                        best_score = score
                        best_content = content
                        best_name = meta["name"]
            except Exception as e:
                print(f"[WARN] 检索过程中无法读取文件 {meta['name']}: {e}")

        if best_content:
            print(
                f"[INFO] 全库相似度检索成功, 最佳匹配: {best_name}, 得分: {best_score:.4f}"
            )
            return best_content

        return ""

    def calculate_tokens(self, text: str) -> int:
        """辅助函数：计算字符串的 Token 数量"""
        return len(self.encoder.encode(text))


if __name__ == "__main__":
    # --- 检索测试脚本 ---
    print("--- 开始测试 ExampleSelector 检索表现 ---")

    # 模拟 Query 代码
    complex_query_code = """
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
    acc = tl.dot(input_data, weight_data, acc)
    tl.store(output_ptrs, acc)
#### END KERNEL
"""

    # 由于 ExampleSelector 默认去本地目录查找，下面的测试逻辑需要确保本地有对应的语料文件
    # 这里仅作为调用演示
    selector = ExampleSelector()

    print("\n[测试 1] 未指定 op_type (触发全库相似度度量):")
    best_any = selector.get_best_example(complex_query_code)
    print("[INFO]********best_anu: ", best_any)
    print("\n[测试 2] 指定 op_type='gemm' (直接采样该类别):")
    best_conv = selector.get_best_example(complex_query_code, op_type="gemm")
    print("[INFO]********best_anu: ", best_conv)

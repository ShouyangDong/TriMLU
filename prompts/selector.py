import os
import tiktoken
import numpy as np
from typing import List, Dict


class ExampleSelector:
    def __init__(self, corpus_dir="prompts/corpus", model_name="gpt-4o"):
        self.corpus_dir = corpus_dir
        # 使用 tiktoken 初始化 encoder
        self.encoder = tiktoken.encoding_for_model(model_name)
        self.examples: List[Dict] = []
        self._load_corpus()

    def _load_corpus(self):
        """加载语料库并预先分词"""
        if not os.path.exists(self.corpus_dir):
            os.makedirs(self.corpus_dir, exist_ok=True)
            return

        files = [f for f in os.listdir(self.corpus_dir) if f.endswith(".py")]
        for f in files:
            file_path = os.path.join(self.corpus_dir, f)
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
                # 提取核心逻辑 (假设你已经有了 extract_kernel 函数)
                kernel_content = self._extract_kernel_logic(content)
                tokens = self.encoder.encode(kernel_content)
                self.examples.append(
                    {
                        "name": f,
                        "content": content,
                        "tokens": set(tokens),  # 使用 set 方便计算重叠度
                        "token_count": len(tokens),
                    }
                )

    def _extract_kernel_logic(self, text: str) -> str:
        """解析 #### START KERNEL 和 #### END KERNEL 之间的内容"""
        import re

        pattern = r"#### START KERNEL(.*?)#### END KERNEL"
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else text

    def get_best_example(self, query_code: str, max_tokens: int = 1000):
        """基于 Token 重叠度（Jaccard Similarity）找出最相关的代码"""
        if not self.examples:
            return ""

        query_tokens = set(self.encoder.encode(query_code))

        best_score = -1.0
        best_content = ""

        for ex in self.examples:
            # 过滤掉太长的 Example，防止超过 Context Window
            if ex["token_count"] > max_tokens:
                continue

            # 计算 Jaccard 相似度: (A ∩ B) / (A ∪ B)
            intersection = query_tokens.intersection(ex["tokens"])
            union = query_tokens.union(ex["tokens"])
            score = len(intersection) / len(union) if union else 0.0

            if score > best_score:
                best_score = score
                best_content = ex["content"]

        return best_content

    def calculate_tokens(self, text: str) -> int:
        """辅助函数：计算字符串的 Token 数量"""
        return len(self.encoder.encode(text))

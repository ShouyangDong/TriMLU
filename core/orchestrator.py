import os
import json
from prompts.templates import (
    get_migrate_prompt,
    get_debug_prompt,
    get_optimize_prompt,
    get_tune_prompt,
)
from prompts.selector import ExampleSelector


class TriMLUOrchestrator:
    def __init__(self, model, kernel_file, output_dir):
        self.model = model
        self.kernel_file = kernel_file
        self.output_dir = output_dir
        self.last_error = ""  # 修复变量未定义问题

        # 初始化 selector
        self.selector = ExampleSelector()

        # 1. 解析核心代码块
        self.kernel_blocks = self._parse_kernel_file(kernel_file)

        # 2. 读取全文代码用于最终替换和写回
        with open(kernel_file, "r") as f:
            self.full_code = f.read()

    def _parse_kernel_file(self, kernel_file):
        """按照标记解析核心 Triton 代码块"""
        blocks = []
        try:
            with open(kernel_file, "r") as file:
                inside_kernel = False
                current_block = []
                for line in file:
                    if line.strip().startswith("#### START KERNEL"):
                        inside_kernel = True
                        current_block = []
                    elif line.strip().startswith("#### END KERNEL"):
                        inside_kernel = False
                        blocks.append("".join(current_block))
                    elif inside_kernel:
                        current_block.append(line)
            print(f"🔍 Extracted {len(blocks)} core kernel blocks.")
        except Exception as e:
            print(f"❌ Error parsing kernel file: {e}")
        return blocks

    def run_pipeline(self, max_iters=3):
        print("🚀 Starting the TriMLU optimization pipeline...")

        # 这里的顺序对应你要求的四个步骤
        steps = ["Migration", "Debugging", "Optimization", "Fine-tuning"]

        for step in steps:
            print(f"🔄 Current Stage: {step}")
            # 针对解析出的每个 kernel block 进行处理
            for idx, block in enumerate(self.kernel_blocks):
                print(f"  ⚡ Processing Block {idx + 1}/{len(self.kernel_blocks)}...")

                # 获取该步骤生成的 response
                response = self._execute_step(step, block, max_iters)

                # 更新 full_code 中的内容（将旧 block 替换为新 code）
                if response and "code" in response:
                    self.full_code = self.full_code.replace(block, response["code"])
                    # 更新 kernel_blocks 列表，确保下一步基于当前最新的代码进行
                    self.kernel_blocks[idx] = response["code"]
                    block = response["code"]  # 为当前循环更新指针

        self._save_results()
        print("✅ Pipeline completed!")

    def _execute_step(self, step, block_code, max_iters):
        """实际调用 LLM 的逻辑，处理不同的 Prompt 模板"""

        if step == "Migration":
            prompt = get_migrate_prompt(block_code)

        elif step == "Debugging":
            # 这里的 last_error 应该来自于你的编译器/执行器
            # 暂时模拟一个错误，实际应调用 self.executor.compile()
            if not self.last_error:
                print("    (Skipping Debug: No error found)")
                return {"code": block_code}
            prompt = get_debug_prompt(block_code, self.last_error)

        elif step == "Optimization":
            # 自动匹配最相似的参考案例
            best_example = self.selector.get_best_example(block_code)
            prompt = get_optimize_prompt(block_code, best_example)

        elif step == "Fine-tuning":
            prompt = get_tune_prompt(block_code)

        else:
            return None

        # 调用模型生成
        response = self.model.generate(prompt)

        # 统一处理返回，确保是 dict
        if isinstance(response, str):
            try:
                response = json.loads(response)
            except:
                # 如果模型没吐 JSON，这里需要做 fallback 处理
                response = {"code": response}

        return response

    def _save_results(self):
        os.makedirs(self.output_dir, exist_ok=True)
        out_path = os.path.join(self.output_dir, os.path.basename(self.kernel_file))
        with open(out_path, "w") as f:
            f.write(self.full_code)
        print(f"💾 Final code saved to: {out_path}")

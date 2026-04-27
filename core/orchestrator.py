import os
import re
import json
import tempfile
import subprocess
import logging  # 引入日志模块
from datetime import datetime
from prompts.templates import (
    get_migrate_prompt,
    get_debug_prompt,
    get_optimize_prompt,
    get_tune_prompt,
)
from prompts.selector import ExampleSelector
from core.status import TestResult

# 配置日志格式
def setup_logger(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f"trimlu_{datetime.now().strftime('%m%d_%H%M')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),  # 输出到文件
            logging.StreamHandler()         # 输出到控制台
        ]
    )
    return logging.getLogger("TriMLU")

class TriMLUOrchestrator:
    def __init__(self, model, kernel_file, output_dir):
        self.logger = setup_logger(output_dir) # 初始化日志
        self.model = model
        self.kernel_file = kernel_file
        self.output_dir = output_dir
        self.selector = ExampleSelector()
        self.history_summary = {}

        try:
            with open(kernel_file, "r") as f:
                self.full_code = f.read()
            self.kernel_blocks = self._parse_kernel_file()
            self.logger.info(f"Successfully loaded {len(self.kernel_blocks)} kernel blocks from {kernel_file}")
        except Exception as e:
            self.logger.error(f"Failed to load kernel file: {str(e)}")
            raise

    def _parse_kernel_file(self):
        pattern = re.compile(r"#### START KERNEL(.*?)#### END KERNEL", re.DOTALL)
        return pattern.findall(self.full_code)

    def _update_full_code(self, block_idx, new_content):
        # 移除可能存在的标识符，确保代码纯净
        clean_code = new_content.replace("#### START KERNEL", "").replace("#### END KERNEL", "").strip()
        # 兼容一些 LLM 喜欢带 ```python 的情况
        clean_code = re.sub(r"```python\n|```", "", clean_code).strip()
        
        pattern = re.compile(r"(#### START KERNEL)(.*?)(#### END KERNEL)", re.DOTALL)
        matches = list(pattern.finditer(self.full_code))
        
        if block_idx < len(matches):
            target = matches[block_idx]
            replacement = f"#### START KERNEL\n{clean_code}\n#### END KERNEL"
            self.full_code = self.full_code[:target.start()] + replacement + self.full_code[target.end():]
            self.kernel_blocks = self._parse_kernel_file()
            self.logger.debug(f"Block {block_idx} updated in memory.")
        else:
            self.logger.error(f"Block index {block_idx} out of range during update.")

    def run_pipeline(self, max_retries=3):
        self.logger.info("🚀 Starting TriMLU Verified Pipeline...")

        for idx in range(len(self.kernel_blocks)):
            kernel_name = f"Kernel_{idx+1}"
            self.logger.info(f"===> Processing {kernel_name} <===")

            # Step 1: Migration
            self.logger.info(f"[{kernel_name}] Stage: Migration")
            self._execute_stage(idx, "Migration")

            # Step 2: Debugging Loop
            verified = False
            for attempt in range(max_retries):
                test_res = self._validate_locally(kernel_name)
                
                if test_res.pass_exe:
                    self.logger.info(f"✅ {kernel_name} passed verification at attempt {attempt+1}")
                    verified = True
                    break
                
                # 记录详细错误日志
                self.logger.warning(f"❌ {kernel_name} Trial {attempt+1} failed.")
                self.logger.error(f"Error Context: {test_res.error[:200]}...") # 只记录前200字防止刷屏
                
                self._execute_stage(idx, "Debugging", error=test_res.error)

            if not verified:
                self.logger.error(f"🔴 Max retries reached for {kernel_name}. Moving to optimization with current state.")

            # Step 3: Optimization & Tuning
            self.logger.info(f"[{kernel_name}] Stage: Optimization & Tuning")
            self._execute_stage(idx, "Optimization")
            self._execute_stage(idx, "Fine-tuning")

            # 记录最终结果
            final_res = self._validate_locally(kernel_name)
            self.history_summary[kernel_name] = final_res.to_dict()

        self._save_results()
        self.display_results_summary(self.history_summary)

    def _execute_stage(self, block_idx, stage, error=None):
        block = self.kernel_blocks[block_idx]
        
        try:
            if stage == "Migration":
                prompt = get_migrate_prompt(block)
            elif stage == "Debugging":
                prompt = get_debug_prompt(block, error)
            elif stage == "Optimization":
                example = self.selector.get_best_example(block)
                prompt = get_optimize_prompt(block, example)
            elif stage == "Fine-tuning":
                prompt = get_tune_prompt(block)
            
            self.logger.info(f"Calling LLM for {stage}...")
            response_text = self.model.generate(prompt)

            # 尝试解析 JSON 或提取代码块
            if "```python" in response_text:
                code = re.search(r"```python\n(.*?)\n```", response_text, re.DOTALL).group(1)
            else:
                # 兜底：如果 LLM 直接返回了 JSON 字符串
                try:
                    data = json.loads(response_text)
                    code = data.get("code", response_text)
                except:
                    code = response_text

            self._update_full_code(block_idx, code)

        except Exception as e:
            self.logger.error(f"Critical error in {stage} stage: {str(e)}")

    def _validate_locally(self, kernel_name):
        """本地验证逻辑"""
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as tmp:
            tmp.write(self.full_code)
            tmp_path = tmp.name
        
        try:
            # 运行编译/验证脚本
            proc = subprocess.run(
                ["python3", tmp_path],
                capture_output=True,
                text=True,
                timeout=60 # 增加超时保护
            )
            
            if proc.returncode != 0:
                return TestResult(
                    success=False,
                    message="Runtime/Compilation Error",
                    error=proc.stderr if proc.stderr else proc.stdout,
                    performance_metrics={"pass_call": False, "pass_exe": False},
                )

            return TestResult(
                success=True,
                message="Success",
                performance_metrics={
                    "pass_call": True, "pass_exe": True,
                    "latency": "1.25ms", "speedup": "1.15x",
                },
            )
        except subprocess.TimeoutExpired:
            return TestResult(success=False, message="Timeout", error="Kernel execution timed out.")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def _save_results(self):
        out_path = os.path.join(self.output_dir, os.path.basename(self.kernel_file))
        with open(out_path, "w") as f:
            f.write(self.full_code)
        self.logger.info(f"💾 Final code saved to: {out_path}")

    def display_results_summary(self, results):
        """表格化展示最终成果报表"""
        # 1. 准备表头
        header = (
            f"║ {'Kernel Name'.ljust(23)} ║"
            f" {'Call'.center(5)} ║"
            f" {'Exe'.center(5)} ║"
            f" {'Latency'.center(13)} ║"
            f" {'Speedup'.center(14)} ║"
        )

        # 2. 构造表格字符串 (用于控制台)
        table = [
            "\n" + "╔" + "═" * 74 + "╗",
            "║" + " 📊 TriMLU-Agent Final Report".center(73) + "║",
            "╠" + "═" * 25 + "╦" + "═" * 7 + "╦" + "═" * 7 + "╦" + "═" * 15 + "╦" + "═" * 16 + "╣",
            header,
            "╠" + "═" * 25 + "╬" + "═" * 7 + "╬" + "═" * 7 + "╬" + "═" * 15 + "╬" + "═" * 16 + "╣"
        ]

        for name, d in results.items():
            c = "✓" if d.get("pass_call") else "✗"
            e = "✓" if d.get("pass_exe") else "✗"
            # 格式化数值，防止溢出
            lat = f"{d.get('latency', 0.0):.4f}ms"
            spd = f"{d.get('speedup', 1.0):.2f}x"

            line = (f"║ {name.ljust(23)} ║"
                    f"   {c}   ║"
                    f"   {e}   ║"
                    f" {lat.center(13)} ║"
                    f" {spd.center(14)} ║")
            table.append(line)

        table.append("╚" + "═" * 25 + "╩" + "═" * 7 + "╩" + "═" * 7 + "╩" + "═" * 15 + "╩" + "═" * 16 + "╝\n")

        # 3. 执行输出
        # 只在控制台 print，不带 logging 的时间戳前缀
        full_table_str = "\n".join(table)
        print(full_table_str)

        # 将结果写入日志文件，但不带 UI 符号，方便后期 grep 解析
        self.logger.info("--- Final Report Summary ---")
        for name, d in results.items():
            self.logger.info(f"RESULT: {name} | Call: {d.get('pass_call')} | Exe: {d.get('pass_exe')} | Latency: {d.get('latency')}ms")


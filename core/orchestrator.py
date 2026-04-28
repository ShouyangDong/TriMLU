import os
import re
import json
import tempfile
import subprocess
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List

from prompts.templates import (
    get_migrate_prompt,
    get_debug_prompt,
    get_optimize_prompt,
    get_tune_prompt,
)
from prompts.selector import ExampleSelector
from core.status import TestResult


# 设置日志记录器
def setup_logger(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(
        output_dir, f"trimlu_{datetime.now().strftime('%m%d_%H%M')}.log"
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger("TriMLU")


class TriMLUOrchestrator:
    def __init__(self, model, kernel_file, output_dir, op_type=None):
        self.logger = setup_logger(output_dir)
        self.model = model
        self.kernel_file = kernel_file
        self.output_dir = output_dir
        self.selector = ExampleSelector()
        self.history_summary = {}
        self.op_type = op_type

        try:
            with open(kernel_file, "r") as f:
                self.full_code = f.read()
            self.kernel_blocks = self._parse_kernel_file()
            self.logger.info(
                f"成功从 {kernel_file} 加载了 {len(self.kernel_blocks)} 个内核块"
            )
        except Exception as e:
            self.logger.error(f"加载内核文件失败: {str(e)}")
            raise

    def _parse_kernel_file(self):
        pattern = re.compile(r"#### START KERNEL(.*?)#### END KERNEL", re.DOTALL)
        return pattern.findall(self.full_code)

    def _update_full_code(self, block_idx, new_content):
        clean_code = (
            new_content.replace("#### START KERNEL", "")
            .replace("#### END KERNEL", "")
            .strip()
        )
        clean_code = re.sub(r"```python\n|```", "", clean_code).strip()

        pattern = re.compile(r"(#### START KERNEL)(.*?)(#### END KERNEL)", re.DOTALL)
        matches = list(pattern.finditer(self.full_code))

        if block_idx < len(matches):
            target = matches[block_idx]
            replacement = f"#### START KERNEL\n{clean_code}\n#### END KERNEL"
            self.full_code = (
                self.full_code[: target.start()]
                + replacement
                + self.full_code[target.end() :]
            )
            self.kernel_blocks = self._parse_kernel_file()
        else:
            self.logger.error(f"更新时索引 {block_idx} 超出范围")

    def run_pipeline(self, max_retries=3):
        self.logger.info("🚀 启动 TriMLU 性能验证流水线 (平均性能版)...")

        for idx in range(len(self.kernel_blocks)):
            kernel_name = f"Kernel_{idx+1}"
            self.logger.info(f"===> 正在处理 {kernel_name} <===")

            # 阶段 1: 基础迁移
            self._execute_stage(idx, "Migration")

            # 阶段 2: 功能验证
            verified = False
            for attempt in range(max_retries):
                test_res = self._validate_locally(kernel_name)
                if test_res.pass_exe:
                    self.logger.info(f"✅ {kernel_name} 通过功能验证")
                    verified = True
                    break

                error_msg = test_res.error if test_res.error else "未知运行错误"
                self.logger.warning(f"❌ {kernel_name} 第 {attempt+1} 次验证失败。")
                self.logger.error(f"错误详情:\n{error_msg}")

                self.logger.info(f"正在进行第 {attempt+1} 次 Debugging...")
                self._execute_stage(idx, "Debugging", error=error_msg)

            if not verified:
                self.logger.error(
                    f"终止处理 {kernel_name}，因为多次尝试后仍未通过验证。"
                )
                continue

            # 阶段 3: 基于平均性能的迭代优化 (无反馈版)
            baseline_res = self._validate_locally(kernel_name)
            best_avg_latency = self._get_avg_latency(baseline_res)
            best_code_state = self.full_code

            self.logger.info(f"📈 基准性能 (平均延迟): {best_avg_latency:.4f} ms")

            for opt_idx in range(max_retries):
                self.logger.info(f"[{kernel_name}] 优化尝试 {opt_idx+1}/{max_retries}")

                # 仅执行优化生成
                self._execute_stage(idx, "Optimization")

                opt_res = self._validate_locally(kernel_name)
                current_avg_latency = self._get_avg_latency(opt_res)

                # 只有在功能正确且性能确实提升时才接受
                if opt_res.pass_exe and current_avg_latency < (
                    best_avg_latency * 0.999
                ):
                    self.logger.info(
                        f"🚀 性能提升: {best_avg_latency:.4f}ms -> {current_avg_latency:.4f}ms"
                    )
                    best_avg_latency = current_avg_latency
                    best_code_state = self.full_code
                else:
                    # 关键修改：如果是代码运行出错，打印具体的错误原因
                    if not opt_res.pass_exe:
                        reason = "代码运行出错"
                        self.logger.warning(f"⚠️ 优化被拒绝 ({reason})")
                        self.logger.error(f"优化版报错详情:\n{opt_res.error}")
                    else:
                        reason = "性能未提升"
                        self.logger.warning(
                            f"⚠️ 优化被拒绝 ({reason}, 当前平均延迟: {current_avg_latency:.4f}ms)"
                        )

                    # 回退到当前最优状态
                    self.full_code = best_code_state
                    self.kernel_blocks = self._parse_kernel_file()

            self.history_summary[kernel_name] = self._validate_locally(
                kernel_name
            ).to_dict()

        self._save_results()
        self.display_results_summary(self.history_summary)

    def _get_avg_latency(self, res: TestResult) -> float:
        """从 TestResult 中安全提取平均延迟数值"""
        if hasattr(res, "latency"):
            if isinstance(res.latency, (int, float)):
                return float(res.latency)
            match = re.findall(r"[\d\.]+", str(res.latency))
            return float(match[0]) if match else float("inf")
        return float("inf")

    def _execute_stage(self, block_idx, stage, error=None):
        block = self.kernel_blocks[block_idx]
        try:
            if stage == "Migration":
                prompt = get_migrate_prompt(block)
            elif stage == "Debugging":
                prompt = get_debug_prompt(block, error)
            elif stage == "Optimization":
                example = self.selector.get_best_example(block, self.op_type)
                prompt = get_optimize_prompt(block, example)

            response_text = self.model.generate(prompt)
            code_match = re.search(r"```python\n(.*?)\n```", response_text, re.DOTALL)
            code = code_match.group(1) if code_match else response_text
            self._update_full_code(block_idx, code)
        except Exception as e:
            self.logger.error(f"{stage} 阶段出错: {str(e)}")

    def _validate_locally(self, kernel_name) -> TestResult:
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as tmp:
            tmp.write(self.full_code)
            tmp_path = tmp.name

        try:
            proc = subprocess.run(
                ["python3", tmp_path], capture_output=True, text=True, timeout=150
            )

            if proc.returncode != 0:
                err_output = proc.stderr if proc.stderr.strip() else proc.stdout
                return TestResult(
                    success=False, message="Runtime Error", error=err_output
                )

            json_match = re.search(r"__TRIMLU_PERF_JSON__:(.*)", proc.stdout)
            latencies = []

            if json_match:
                try:
                    data = json.loads(json_match.group(1))
                    items = data if isinstance(data, list) else [data]
                    latencies = [
                        float(item.get("latency", item.get("triton_ms", 0.0)))
                        for item in items
                    ]
                except:
                    pass

            if not latencies:
                lat_match = re.search(r"triton:\s*([\d\.]+)\s*ms", proc.stdout)
                if lat_match:
                    latencies = [float(lat_match.group(1))]

            avg_lat = sum(latencies) / len(latencies) if latencies else 0.0

            metrics = {
                "pass_call": True,
                "pass_exe": True,
                "latency": avg_lat,
                "count": len(latencies),
            }

            return TestResult(
                success=True,
                message="Success",
                performance_metrics=metrics,
                execution_time=avg_lat,
            )

        except subprocess.TimeoutExpired:
            return TestResult(
                success=False, message="Timeout", error="进程运行超时 (150s)"
            )
        except Exception as e:
            return TestResult(
                success=False, message="Validation Exception", error=str(e)
            )
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def _save_results(self):
        out_path = os.path.join(self.output_dir, os.path.basename(self.kernel_file))
        with open(out_path, "w") as f:
            f.write(self.full_code)

    def display_results_summary(self, results):
        print("\n" + "=" * 60)
        print("📊 TriMLU 最终性能汇总 (平均值)".center(60))
        print("=" * 60)
        for k, v in results.items():
            status = "✅ 成功" if v.get("pass_exe") else "❌ 失败"
            print(f"名称: {k}")
            print(f"  - 状态: {status}")
            print(f"  - 平均耗时: {v.get('latency', 'N/A')} ms")
            print(f"  - 测试用例数: {v.get('count', 0)}")
        print("=" * 60 + "\n")

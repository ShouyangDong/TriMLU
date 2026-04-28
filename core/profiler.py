import os
import re
import json
import subprocess
import tempfile
import pandas as pd
from typing import Dict, List, Any


class MLUProfiler:
    """
    Wrapper for Cambricon cnperf-cli to profile MLU kernels.
    """

    def __init__(self, output_dir: str = "profiler_logs"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        # Standard events for bottleneck analysis
        self.default_events = [
            "tp_core__lt_cycles",  # Logic Timer Cycles
            "tp_core__ct_cycles",  # Compute Timer Cycles
            "tp_core__lram_bank_conflict",  # Bank Conflicts
            "tp_core__instr_stall_data",  # Stalls due to data dependency
        ]

    def profile_kernel(self, full_code: str, kernel_name: str) -> Dict:
        """
        统一入口：Orchestrator 调用此方法进行硬件性能分析。
        1. 将代码写入临时文件
        2. 运行 cnperf-cli 录制
        3. 导出并分析数据
        """
        print(f"🔍 [Profiler] 正在分析 {kernel_name} 的硬件瓶颈...")

        # 1. 准备临时运行文件
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as tmp:
            tmp.write(full_code)
            tmp_path = tmp.name

        try:
            # 2. 录制数据 (.cnperf-rep)
            rep_file = self.record_kernel(["python3", tmp_path])

            # 3. 导出 CSV
            csv_dir = self.export_to_csv(rep_file)

            # 4. 分析瓶颈指标
            analysis_results = self.analyze_bottlenecks(csv_dir)

            return analysis_results

        except Exception as e:
            print(f"⚠️ [Profiler] 分析失败: {str(e)}")
            return {"error": str(e)}
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def record_kernel(self, program_cmd: List[str], events: List[str] = None) -> str:
        """
        Executes cnperf-cli record to capture PMU data.
        """
        event_str = ",".join(events or self.default_events)
        output_name = os.path.join(self.output_dir, "perf_data.cnperf-rep")

        cmd = [
            "cnperf-cli",
            "record",
            "--pmu",
            "true",
            "--events",
            event_str,
            "-o",
            output_name,
            "-f",  # Force overwrite
        ] + program_cmd

        print(f"[Profiler] Recording: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, capture_output=True)
        return output_name

    def export_to_csv(self, rep_file: str) -> str:
        """
        Parses the binary .cnperf-rep file into a CSV folder.
        """
        csv_dir = os.path.join(self.output_dir, "csv_results")
        cmd = ["cnperf-cli", "kernel", "--csv", "-o", csv_dir, "-f", rep_file]
        print(f"[Profiler] Exporting to CSV: {csv_dir}")
        subprocess.run(cmd, check=True, capture_output=True)
        return csv_dir

    def analyze_bottlenecks(self, csv_dir: str) -> Dict:
        """
        Reads the generated CSV and extracts key diagnostic metrics.
        """
        pmu_csv = os.path.join(csv_dir, "kernel_pmu.csv")
        if not os.path.exists(pmu_csv):
            # Fallback for different versions/naming
            files = [f for f in os.listdir(csv_dir) if f.endswith(".csv")]
            if not files:
                return {"error": "No CSV files found"}
            pmu_csv = os.path.join(csv_dir, files[0])

        try:
            df = pd.read_csv(pmu_csv)

            # 基础指标统计
            summary = {
                "total_kernels": len(df),
                "avg_lt_cycles": float(df.get("tp_core__lt_cycles", [0]).mean()),
                "avg_ct_cycles": float(df.get("tp_core__ct_cycles", [0]).mean()),
                "bank_conflicts": int(df.get("tp_core__lram_bank_conflict", [0]).sum()),
                "data_stalls": int(df.get("tp_core__instr_stall_data", [0]).sum()),
            }

            # 计算计算效率比例
            if summary["avg_lt_cycles"] > 0:
                summary["compute_efficiency"] = round(
                    summary["avg_ct_cycles"] / summary["avg_lt_cycles"], 4
                )

            return summary
        except Exception as e:
            return {"error": f"CSV解析失败: {str(e)}"}

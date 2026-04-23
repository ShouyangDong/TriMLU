class TestResult:
    """封装编译和运行测试的结果"""

    def __init__(
        self,
        success,
        message,
        execution_time=None,
        performance_metrics=None,
        error=None,
    ):
        self.success = success
        self.message = message
        self.execution_time = execution_time
        self.performance_metrics = performance_metrics or {}
        self.error = error

    def to_dict(self):
        return {
            "pass_call": self.pass_call,
            "pass_exe": self.pass_exe,
            "pass_perf": self.pass_perf,
            "latency": f"{self.latency:.4f}ms",
            "speedup": f"{self.speedup:.2x}",
        }

    def __str__(self):
        status = "✅ PASS" if self.success else "❌ FAIL"
        res = f"[{status}] {self.message}"
        if self.error:
            res += f"\n   Error Details: {self.error[:200]}..."
        return res


def display_results_summary(results, iteration=None):
    """显示格式化的测试结果表格"""
    if not results:
        print("⚠️ No results found. Run the agent first.")
        return

    # 定义颜色 (可选，如果在终端运行)
    GREEN = "\033[92m"
    RED = "\033[91m"
    RESET = "\033[0m"

    print("\n" + "╔" + "═" * 78 + "╗")
    print("║" + " 📊 TriMLU-Agent Optimization Report".center(78) + "║")
    if iteration:
        print("║" + f" Iteration: {iteration}".center(78) + "║")
    print(
        "╠"
        + "═" * 25
        + "╦"
        + "═" * 6
        + "╦"
        + "═" * 6
        + "╦"
        + "═" * 6
        + "╦"
        + "═" * 15
        + "╦"
        + "═" * 11
        + "╣"
    )
    print(
        "║"
        + " Kernel Name".ljust(25)
        + "║"
        + " Call ".center(6)
        + "║"
        + " Exe ".center(6)
        + "║"
        + " Perf ".center(6)
        + "║"
        + " Latency".center(15)
        + "║"
        + " Speedup ".center(11)
        + "║"
    )
    print(
        "╠"
        + "═" * 25
        + "╬"
        + "═" * 6
        + "╬"
        + "═" * 6
        + "╬"
        + "═" * 6
        + "╬"
        + "═" * 15
        + "╬"
        + "═" * 11
        + "╣"
    )

    for name, data in results.items():
        # 转换符号
        c = "✓" if data.get("pass_call") else "✗"
        e = "✓" if data.get("pass_exe") else "✗"
        p = "✓" if data.get("pass_perf") else "✗"

        lat = str(data.get("latency", "N/A"))
        spd = str(data.get("speedup", "N/A"))

        # 格式化行
        row = f"║ {name.ljust(23)} ║  {c}   ║  {e}   ║  {p}   ║ {lat.center(13)} ║ {spd.center(9)} ║"
        print(row)

    print(
        "╚"
        + "═" * 25
        + "╩"
        + "═" * 6
        + "╩"
        + "═" * 6
        + "╩"
        + "═" * 6
        + "╩"
        + "═" * 15
        + "╩"
        + "═" * 11
        + "╝\n"
    )

class TestResult:
    """封装编译和运行测试的结果"""

    def __init__(
        self,
        success: bool,
        message: str,
        execution_time=None,
        performance_metrics=None,
        error=None,
    ):
        self.success = success
        self.message = message
        self.execution_time = execution_time
        self.performance_metrics = performance_metrics or {}
        self.error = error

        # 显式保留 6 个关键状态属性，方便直接访问
        self.pass_call = self.performance_metrics.get("pass_call", False)
        self.pass_exe = self.performance_metrics.get("pass_exe", False)
        self.pass_perf = self.performance_metrics.get("pass_perf", False)

        latency = self.performance_metrics.get("latency", 0)
        speedup = self.performance_metrics.get("speedup", 1.0)

        self.latency = (
            f"{latency:.4f}ms" if isinstance(latency, (int, float)) else "N/A"
        )
        self.speedup = f"{speedup:.2f}x" if isinstance(speedup, (int, float)) else "N/A"

    def to_dict(self):
        return {
            "pass_call": self.pass_call,
            "pass_exe": self.pass_exe,
            "pass_perf": self.pass_perf,
            "latency": self.latency,
            "speedup": self.speedup,
        }

    def __str__(self):
        status = "✅ PASS" if self.success else "❌ FAIL"
        res = f"[{status}] {self.message}"
        if self.error:
            res += f"\n   Error Details: {self.error[:200]}..."
        return res

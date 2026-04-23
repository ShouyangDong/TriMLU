import os
import sys
import argparse
from typing import Optional

# ═══════════════════════════════════════════════════════════════
# 1. 内部执行逻辑
# ═══════════════════════════════════════════════════════════════
def run_trimlu_optimization(
    kernel_file: str,
    output_dir: str = "outputs",
    iteration_num: int = 3,
    target_mlu: str = "MLU590",
    model_id: str = "gpt-4o",
    api_key: Optional[str] = None,
    verbose: bool = True,
):
    from core.orchestrator import TriMLUOrchestrator
    from core.llm_model import OpenAIModel
    from utils import print_header, print_config # 假设你有一套 UI 工具

    if verbose:
        print_header("✓ TriMLU Engine Ready")

    # 配置 API 和模型
    final_api_key = api_key or os.getenv("OPENAI_API_KEY")
    model = OpenAIModel(api_key=final_api_key, model_id=model_id)

    # 初始化调度器 (这是核心逻辑所在)
    # 注意：这里只传文件名，读取逻辑在内部实现
    orchestrator = TriMLUOrchestrator(
        model=model,
        kernel_file=kernel_file,
        output_dir=output_dir
    )

    if verbose:
        print(f"🔍 正在加载 Kernel: {kernel_file} ...")
        print_config({"target": target_mlu, "iters": iteration_num}, "🚀 开始迁移与优化")

    # 执行四步走策略
    try:
        # 1. 迁移 -> 2. Debug -> 3. 优化 -> 4. 调优
        orchestrator.run_pipeline(max_iters=iteration_num)
    except Exception as e:
        print(f"❌ 运行失败: {e}")
        return None

    print(f"\n✅ 迁移优化完成! 结果保存在: {output_dir}")
    return output_dir

# ═══════════════════════════════════════════════════════════════
# 2. 命令行参数解析
# ═══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="🚀 TriMLU GPU-to-MLU 优化工具")
    
    # 必需参数：只给文件名
    parser.add_argument("kernel_file", type=str, help="待优化的 kernel 文件名")

    # 可选参数
    parser.add_argument("-o", "--output", default="outputs", help="输出目录")
    parser.add_argument("-i", "--iterations", type=int, default=3, help="最大修复/优化迭代次数")
    parser.add_argument("--model", default="gpt-4o", help="LLM 模型 ID")
    parser.add_argument("--target", default="MLU590", help="目标硬件型号")
    parser.add_argument("--api-key", help="API Key (可选)")

    args = parser.parse_args()

    run_trimlu_optimization(
        kernel_file=args.kernel_file,
        output_dir=args.output,
        iteration_num=args.iterations,
        target_mlu=args.target,
        model_id=args.model,
        api_key=args.api_key
    )

if __name__ == "__main__":
    main()

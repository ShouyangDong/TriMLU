from core.orchestrator import TriMLUOrchestrator
from core.models import OpenAIModel, ClaudeModel, GeminiModel
from core.utils import print_header, print_config
import os
import argparse
import sys


def run_trimlu_optimization(
    kernel_file: str,
    model_type: str = "openai",  # 默认调用 openai，可选: openai, claude, gemini
    model_id: str = None,
    output_dir: str = "outputs",
    iteration_num: int = 3,
    target_mlu: str = "MLU590",
    api_key: str = None,
    azure_endpoint: str = None,
    verbose: bool = True,
):
    if verbose:
        print_header("✓ TriMLU Engine Ready")

    # 根据模型类型配置 API 和初始化模型
    model = None
    if model_type == "openai":
        final_api_key = api_key or os.getenv("OPENAI_API_KEY")
        final_azure_endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        model = OpenAIModel(
            model_id=model_id,
            api_key=final_api_key,
            azure_endpoint=final_azure_endpoint,
        )
    elif model_type == "claude":
        model_id = model_id if (model_id and model_id.startswith("claude")) else "claude-sonnet-4-5-20250929"
        final_api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        model = ClaudeModel(
            model_id=model_id,
            api_key=final_api_key,
        )
    elif model_type == "gemini":
        final_api_key = api_key or os.getenv("GOOGLE_API_KEY")
        model = GeminiModel(
            model_id=model_id,
            api_key=final_api_key,
        )
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

    # 初始化调度器
    orchestrator = TriMLUOrchestrator(
        model=model, kernel_file=kernel_file, output_dir=output_dir
    )

    if verbose:
        print(f"🔍 正在加载 Kernel: {kernel_file} (使用模型: {model_type}) ...")
        print_config(
            {
                "target": target_mlu,
                "iters": iteration_num,
                "model": model_id,
                "provider": model_type,
            },
            "🚀 开始迁移与优化",
        )

    # 执行四步走策略
    try:
        # 注意：Orchestrator 内部调用的方法名应与类中一致
        orchestrator.run_pipeline(max_retries=iteration_num)
    except Exception as e:
        print(f"❌ 运行失败: {e}")
        import traceback

        if verbose:
            traceback.print_exc()
        return None

    print(f"\n✅ 迁移优化完成! 结果保存在: {output_dir}")
    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="TriMLU: 寒武纪 MLU Triton 算子自动迁移与优化工具"
    )

    # 核心参数
    parser.add_argument(
        "kernel_file", type=str, help="待处理的原始 Triton Kernel 文件路径 (.py)"
    )

    # 模型配置
    parser.add_argument(
        "--model-type",
        type=str,
        default="openai",
        choices=["openai", "claude", "gemini"],
        help="使用的模型提供商 (默认: openai)",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="gpt-4",
        help="具体模型 ID (如 gpt-4, claude-sonnet-4-5-20250929, gemini-1.5-pro)",
    )

    # 运行配置
    parser.add_argument(
        "--output-dir", type=str, default="outputs", help="结果输出目录"
    )
    parser.add_argument(
        "--iters", type=int, default=3, help="最大尝试重试/优化迭代次数"
    )
    parser.add_argument(
        "--target", type=str, default="MLU590", help="目标硬件平台 (默认: MLU590)"
    )

    # 认证配置 (通常推荐通过环境变量，但也支持命令行传入)
    parser.add_argument(
        "--api-key", type=str, default=None, help="API Key (若不提供则从环境变量读取)"
    )
    parser.add_argument(
        "--endpoint", type=str, default=None, help="Azure Endpoint (仅 OpenAI 模式需要)"
    )

    # 其他
    parser.add_argument("--quiet", action="store_true", help="关闭详细日志输出")

    args = parser.parse_args()

    # 检查文件是否存在
    if not os.path.exists(args.kernel_file):
        print(f"❌ 错误: 找不到输入文件 '{args.kernel_file}'")
        sys.exit(1)

    # 启动优化
    run_trimlu_optimization(
        kernel_file=args.kernel_file,
        model_type=args.model_type,
        model_id=args.model_id,
        output_dir=args.output_dir,
        iteration_num=args.iters,
        target_mlu=args.target,
        api_key=args.api_key,
        azure_endpoint=args.endpoint,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()

from core.orchestrator import TriMLUOrchestrator
from core.llm_model import OpenAIModel
from utils import print_header, print_config
import os


def run_trimlu_optimization(
    kernel_file: str,
    output_dir: str = "outputs",
    iteration_num: int = 3,
    target_mlu: str = "MLU590",
    model_id: str = "gpt-4",
    api_key: str = None,
    azure_endpoint: str = None,
    verbose: bool = True,
):
    if verbose:
        print_header("✓ TriMLU Engine Ready")

    # 配置 API 和模型
    final_api_key = api_key or os.getenv("OPENAI_API_KEY")
    final_azure_endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
    model = OpenAIModel(
        model_id=model_id,
        api_key=final_api_key,
        azure_endpoint=final_azure_endpoint,
    )

    # 初始化调度器
    orchestrator = TriMLUOrchestrator(
        model=model, kernel_file=kernel_file, output_dir=output_dir
    )

    if verbose:
        print(f"🔍 正在加载 Kernel: {kernel_file} ...")
        print_config(
            {"target": target_mlu, "iters": iteration_num}, "🚀 开始迁移与优化"
        )

    # 执行四步走策略
    try:
        orchestrator.run_pipeline(max_iters=iteration_num)
    except Exception as e:
        print(f"❌ 运行失败: {e}")
        return None

    print(f"\n✅ 迁移优化完成! 结果保存在: {output_dir}")
    return output_dir

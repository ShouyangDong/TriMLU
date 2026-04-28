from core.orchestrator import TriMLUOrchestrator
from core.models import OpenAIModel, ClaudeModel, GeminiModel
from core.utils import print_header, print_config
import os
import argparse
import sys


def run_trimlu_optimization(
    kernel_file: str,
    model_type: str = "openai",
    model_id: str = None,
    op_type: str = None,
    output_dir: str = "outputs",
    iteration_num: int = 3,
    target_mlu: str = "MLU590",
    api_key: str = None,
    azure_endpoint: str = None,
    verbose: bool = True,
):

    if verbose:
        print_header("✓ TriMLU Engine Ready")

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
        model_id = (
            model_id
            if (model_id and model_id.startswith("claude"))
            else "claude-sonnet-4-5-20250929"
        )
        final_api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        model = ClaudeModel(model_id=model_id, api_key=final_api_key)
    elif model_type == "gemini":
        final_api_key = api_key or os.getenv("GOOGLE_API_KEY")
        model = GeminiModel(model_id=model_id, api_key=final_api_key)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    orchestrator = TriMLUOrchestrator(
        model=model, kernel_file=kernel_file, output_dir=output_dir, op_type=op_type
    )

    if verbose:
        print(f"🔍 Loading Kernel: {kernel_file} (Model: {model_type}) ...")
        print_config(
            {
                "target": target_mlu,
                "iters": iteration_num,
                "model": model_id,
                "provider": model_type,
                "op_type": op_type or "Auto (Similarity)",
            },
            "🚀 Starting Migration & Optimization",
        )

    try:
        orchestrator.run_pipeline(max_retries=iteration_num)
    except Exception as e:
        print(f"❌ Execution Failed: {e}")
        if verbose:
            import traceback

            traceback.print_exc()
        return None

    print(f"\n✅ Migration & Optimization Completed! Results saved in: {output_dir}")
    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="TriMLU: Automated Triton Kernel Migration and Optimization for Cambricon MLU"
    )

    parser.add_argument(
        "kernel_file", type=str, help="Path to the original Triton Kernel file (.py)"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="openai",
        choices=["openai", "claude", "gemini"],
        help="LLM Provider",
    )
    parser.add_argument(
        "--model-id", type=str, default="gpt-4o", help="Specific Model ID"
    )
    parser.add_argument(
        "--op-type",
        type=str,
        default=None,
        help="Operator type (e.g., gemm, elewise). Auto-detects via similarity if not provided.",
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs", help="Output directory"
    )
    parser.add_argument(
        "--iters", type=int, default=3, help="Max iteration/retry count"
    )
    parser.add_argument(
        "--target", type=str, default="MLU590", help="Target hardware platform"
    )
    parser.add_argument("--api-key", type=str, default=None, help="API Key")
    parser.add_argument(
        "--endpoint", type=str, default=None, help="Azure Endpoint (OpenAI only)"
    )
    parser.add_argument("--quiet", action="store_true", help="Disable verbose logging")

    args = parser.parse_args()

    if not os.path.exists(args.kernel_file):
        print(f"❌ Error: Input file '{args.kernel_file}' not found.")
        sys.exit(1)

    run_trimlu_optimization(
        kernel_file=args.kernel_file,
        model_type=args.model_type,
        model_id=args.model_id,
        op_type=args.op_type,
        output_dir=args.output_dir,
        iteration_num=args.iters,
        target_mlu=args.target,
        api_key=args.api_key,
        azure_endpoint=args.endpoint,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()

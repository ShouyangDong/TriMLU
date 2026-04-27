def print_header(title):
    """打印醒目的阶段标题"""
    print("\n" + "=" * 50)
    print(f"{title:^50}")
    print("=" * 50)


def print_config(config, title="配置信息"):
    """打印参数配置列表"""
    print(f"\n[ {title} ]")
    for key, value in config.items():
        print(f"  > {key:.<20} : {value}")


def display_optimized_kernel(kernel_name, code_block):
    """
    专门用于展示优化后的 Kernel 代码
    带有代码高亮风格的边框和函数签名提取
    """
    import re

    print_header(f"✨ 优化结果: {kernel_name}")

    # 尝试提取函数签名
    signature = re.search(r"def\s+([a-zA-Z_].*?)\s*:", code_block)
    if signature:
        print(f"📌 函数签名: {signature.group(1)}")

    print("\n--- Kernel Code Start ---")
    print(code_block.strip())
    print("--- Kernel Code End ---\n")

    # 打印一些统计建议（示例）
    print("💡 优化建议: 检查存储掩码以减少不必要的内存写回。")
    print("=" * 50 + "\n")


# 使用示例 (仅用于本地测试):
if __name__ == "__main__":
    test_code = """
@triton.jit
def softmax_kernel_v2(output_ptr, input_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
    # 优化后的持久化循环逻辑...
    pass
    """

    config_data = {
        "算子名称": "Softmax",
        "优化策略": "Persistent Kernel + Vectorized Load",
        "目标硬件": "MLU590",
        "预期加速比": "1.45x",
    }

    print_config(config_data, "优化任务概览")
    display_optimized_kernel("Softmax_Persistent_V2", test_code)

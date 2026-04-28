import json

# ==============================================================================
# MLU 核心约束
# ==============================================================================
MLU_HARDWARE_CONSTRAINTS = """
- **NRAM Aware**: Keep BLOCK_SIZE power-of-two (64, 128, 256).
- **Type Promotion**: Use `.to(tl.float32)` for accumulations. Inputs for `tl.dot` must be float16/bfloat16.
- **Strict API**: NO `tl.pointer`. Use `tl.math` only. 
- **Math Fallback**: If `tl.math.tanh` is missing, use (tl.math.exp(2*x)-1)/(tl.math.exp(2*x)+1).
- **Grid Match**: tl.program_id axis must strictly match the launch grid dimensions.
"""


# 封装一个辅助函数，统一转换为 API 所需列表格式
def format_as_user_msg(prompt_text):
    return [{"role": "user", "content": prompt_text.strip()}]


# 1. 迁移阶段
def get_migrate_prompt(current_code):
    prompt = f"""
You are an expert Triton-MLU Compiler Engineer. 
Migrate the following GPU Triton code to Cambricon MLU590.

### MLU CONSTRAINTS:
{MLU_HARDWARE_CONSTRAINTS}

### INPUT CODE:
{current_code}

### OUTPUT INSTRUCTIONS:
Directly provide the migrated Python code within a ```python ... ``` code block. 
NO explanations, NO JSON, and NO text before or after the code block.
"""
    return format_as_user_msg(prompt)


# 2. 调试阶段
def get_debug_prompt(current_code, error_log):
    prompt = f"""
You are an MLU Kernel Debugger. Analyze the error and fix the code.

### ERROR LOG:
{error_log}

### MLU CONSTRAINTS:
{MLU_HARDWARE_CONSTRAINTS}

### CURRENT CODE:
{current_code}

### OUTPUT INSTRUCTIONS:
Directly provide the fixed Python code within a ```python ... ``` code block. 
NO explanations, NO JSON, and NO text before or after the code block.
"""
    return format_as_user_msg(prompt)


# 3. 优化阶段
def get_optimize_prompt(current_code, example_code=""):
    example_section = f"### REFERENCE PATTERN:\n{example_code}" if example_code else ""

    prompt = f"""
You are a Cambricon MLU Performance Engineer. 
Optimize the following kernel for MLU590 using algorithmic fusion and memory coalescing.

{example_section}

### OPTIMIZATION GOALS:
- Maximize NRAM reuse.
- Implement software pipelining.
- Use efficient algorithmic variants (e.g., Online Softmax).

### CURRENT CODE:
{current_code}

### OUTPUT INSTRUCTIONS:
Directly provide the optimized Python code within a ```python ... ``` code block. 
NO explanations, NO JSON, and NO text before or after the code block.
"""
    return format_as_user_msg(prompt)


# 4. 调优阶段
def get_tune_prompt(current_code, example=None):
    """
    Generate prompt for hardware-level tuning.
    :param current_code: The kernel code to be tuned.
    :param example: Optional reference example showing autotune patterns.
    """
    example_section = ""
    if example:
        example_section = f"""
### REFERENCE EXAMPLE (Best Practice):
{example}
"""

    prompt = f"""
You are a Hardware Tuning Expert.
Inject `@triton.autotune` into the following kernel to maximize throughput on MLU hardware.

### TUNING PARAMETERS:
- Identify the tiling parameters used in the kernel (e.g., BLOCK_SIZE, BLOCK_M, BLOCK_N, BLOCK_K, etc.).
- Extract and define appropriate search spaces for these parameters based on the reference example, hardware constraints, and kernel complexity.
- Define a suitable range for `num_stages` (e.g., [1, 2, 3, 4]) based on the memory constraints and pipeline requirements seen in the reference example.
- num_warps: [1, 4, 8]

{example_section}

### CURRENT CODE:
{current_code}

### OUTPUT INSTRUCTIONS:
Directly provide the full Python code (including the @triton.autotune decorator, configs list, and the kernel) within a ```python ... ``` code block.
Ensure that all necessary hyperparameters are passed as `tl.constexpr` to the kernel.
NO explanations, NO JSON, and NO text before or after the code block.
"""
    return format_as_user_msg(prompt)

import json

# ==============================================================================
# MLU 核心约束：通过显式列表强化 LLM 的硬件意识
# ==============================================================================
MLU_HARDWARE_CONSTRAINTS = """
- **NRAM Aware**: Keep BLOCK_SIZE power-of-two (64, 128, 256).
- **Type Promotion**: Use `.to(tl.float32)` for accumulations. Inputs for `tl.dot` must be float16/bfloat16.
- **Strict API**: NO `tl.pointer`. Use `tl.math` only. 
- **Math Fallback**: If `tl.math.tanh` is missing, use `(tl.math.exp(2*x)-1)/(tl.math.exp(2*x)+1)`.
- **Grid Match**: `tl.program_id` axis must strictly match the launch grid dimensions.
"""


# 1. 迁移阶段
def get_migrate_prompt(current_code):
    return f"""
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


# 2. 调试阶段
def get_debug_prompt(current_code, error_log):
    return f"""
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


# 3. 优化阶段
def get_optimize_prompt(current_code, example_code=""):
    example_section = f"### REFERENCE PATTERN:\n{example_code}" if example_code else ""

    return f"""
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


# 4. 调优阶段
def get_tune_prompt(current_code):
    return f"""
You are a Hardware Tuning Expert. 
Inject `@triton.autotune` into the following kernel.

### TUNING PARAMETERS:
- BLOCK_M/N/K: [32, 64, 128, 256]
- num_stages: [1, 2, 3, 4, 5]
- num_warps: [1, 4]

### CURRENT CODE:
{current_code}

### OUTPUT INSTRUCTIONS:
Directly provide the full Python code (including decorator and configs) within a ```python ... ``` code block. 
NO explanations, NO JSON, and NO text before or after the code block.
"""

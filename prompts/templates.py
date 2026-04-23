import json


# 1. 迁移阶段
def get_migrate_prompt(current_code):
    return f"""
### ROLE:
You are a Triton Compiler Expert.

### TASK:
Migrate the following GPU Triton code to Cambricon MLU.
1. Adjust hardware-specific hints for MLU590.
2. Ensure NRAM compatibility.

### TARGET CODE:
{current_code}

### OUTPUT FORMAT:
Return a JSON object with "strategy" and "code".
"""


# 2. 调试阶段
def get_debug_prompt(current_code, error_log):
    return f"""
### ROLE:
You are an MLU Kernel Debugger.

### ERROR LOG:
{error_log}

### TASK:
Fix the compilation errors in the following code.
Target Code:
{current_code}

### OUTPUT FORMAT:
Return a JSON object with "error_analysis" and "code".
"""


# 3. 优化阶段
def get_optimize_prompt(current_code, example_code=""):
    example_section = ""
    if example_code:
        example_section = f"""
### REFERENCE EXAMPLE OF OPTIMIZED MLU KERNEL:
Below is an example of a high-performance MLU kernel for a similar task. 
Use its optimization patterns (e.g., NRAM alignment, pipelining) as a reference:
```python
{example_code}
"""  # <--- 确保这一行闭合了
    return f"""
ROLE:
You are a Cambricon MLU Performance Engineer.
{example_section}

TASK:
Optimize the following Triton kernel block for MLU hardware.
Current Code:
{current_code}

OUTPUT FORMAT:
Return a JSON object with "reasoning" and "code".
"""


def get_tune_prompt(current_code):
    return f"""ROLE:
You are a Hardware Tuning Expert.

TASK:
Suggest a @triton.autotune search space for this MLU kernel.  # <--- 这里加上 @ 符号
Code:
{current_code}

OUTPUT FORMAT:
Return a JSON object with "configs" (list of strings) and "code" (full code with decorator).
"""

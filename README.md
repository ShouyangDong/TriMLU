🚀 TriMLU-Agent 使用指南

TriMLU-Agent 是一个专为 寒武纪 MLU 设计的 Triton 算子自动化迁移与优化工具。它利用 LLM（大语言模型）的推理能力，自动将 NVIDIA GPU 上的 Triton 代码转换为高效的 MLU 内核代码。

🛠 1. 环境准备

在使用之前，请确保你的开发环境满足以下条件：

Python 环境: Python 3.9+。

依赖库:

pip install triton  # 确保是适配 MLU 的版本
pip install pytest  # 用于运行测试用例


配置文件:

确保你的模型 API Key 已配置在环境变量中。

📂 2. 代码标注规范

为了让 Agent 精准识别需要处理的代码块，你需要在原始 Python 文件中使用特殊的注释标记。

标记示例 (my_kernel.py):

import triton
import triton.language as tl

### 标记外的内容（如 import）会被原样保留
\#### START KERNEL

@triton.jit
def matmul_kernel(...):
    # 这里是待处理的核心代码
    ...

\#### END KERNEL


⚡ 3. 运行转换 (Execute Conversion)

准备好待迁移的文件后，通过项目根目录下的 run_client.py 脚本启动自动化流水线：

## 基础运行命令
python3 run_client.py your_kernel_file.py

## 示例：迁移并优化当前目录下的 my_kernel.py
python3 run_client.py my_kernel.py


执行参数说明：

第一个参数：待处理的 .py 文件路径。

该脚本会自动调用 TriMLUOrchestrator，并在当前目录下查找 #### START KERNEL 标记的代码块。

🔄 4. 自动化流程 (The Pipeline)

运行后，每个代码块会顺序经历以下四个阶段：

第一阶段：迁移 (Migration)

目标: 将 GPU 特有的语法转换为 MLU 语法。

核心动作: 调整数据类型（如使用 tl.float32 累加）、适配 MLU 指令集、处理 tl.math 库差异。

第二阶段：调试 (Auto-Debugging)

动作:

Agent 将中间代码保存为临时文件。

调用 subprocess 进行编译和语法检查。

若失败，捕获 Error Log 自动回传给 LLM 进行逻辑修复。

第三阶段：优化 (Optimization)

目标: 硬件性能压榨。

动作: 应用 NRAM 对齐、双缓冲、算子融合 等策略。

第四阶段：调优 (Fine-tuning)

目标: 自动化搜索最优超参数。

动作: 注入 @triton.autotune，并预设符合 MLU 硬件特性的 num_stages 和 num_warps 搜索空间。

📊 5. 结果展示与导出

任务完成后，Agent 会执行以下操作：

导出文件: 在 test_outputs/ 目录下生成同名文件。你可以对比生成的代码与原代码的差异。

可视化报告: 在终端输出一份 ASCII 表格报表，展示各 Kernel 的编译情况、延迟 (Latency) 及加速比 (Speedup)。

🧪 6. 运行系统测试

验证 TriMLU 本身的各项组件是否正常：

## 1. 运行 Prompt 模板测试（检查 LLM 指令是否正确生成）
python3 -m unittest tests.test_templates

## 2. 运行编排引擎集成测试（检查代码块替换逻辑）
python3 -m unittest tests.test_orchestrator

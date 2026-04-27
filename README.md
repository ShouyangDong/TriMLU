# 🚀 TriMLU-Agent 使用指南

TriMLU-Agent 是一个专为 **寒武纪 MLU** 设计的 Triton 算子自动化迁移与优化工具。它利用 LLM（大语言模型）的推理能力，自动将 NVIDIA GPU 上的 Triton 代码转换为高效的 MLU 内核代码。

## 🛠 1. 环境准备

在使用之前，请确保你的开发环境满足以下条件：

1.  **Python 环境**: Python 3.9+。
2.  **依赖库**:
    ```bash
    pip install triton  # 确保是适配 MLU 的版本
    pip install pytest  # 用于运行单元测试
    pip install openai anthropic google-generativeai tenacity  # 模型驱动依赖
    ```
3.  **API 配置**:
    根据你选择的模型，在环境变量中配置对应的 Key：
    - **OpenAI/Azure**: `OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`
    - **Claude**: `ANTHROPIC_API_KEY`
    - **Gemini**: `GOOGLE_API_KEY`

## 📂 2. 代码标注规范

为了让 Agent 精准识别需要处理的代码块，你需要在原始 Python 文件中使用特殊的注释标记。

**标记示例 (`my_kernel.py`):**

```python
import triton
import triton.language as tl

# 标记外的内容（如 import）会被原样保留
#### START KERNEL
@triton.jit
def matmul_kernel(A, B, C, BLOCK_SIZE: tl.constexpr):
    # 这里是待处理的核心代码
    ...
#### END KERNEL
```

## ⚡ 3. 运行转换 (Execute Conversion)

你可以通过 `run_client.py` 脚本启动自动化流水线，并根据需要切换不同的 LLM 后端。

### 基础运行 (默认使用 OpenAI)
```bash
python3 run_client.py my_kernel.py
```

### 切换模型后端 (Claude 或 Gemini)
```bash
# 使用 Claude 3.5 Sonnet
python3 run_client.py my_kernel.py --model-type claude --model-id claude-3-5-sonnet-20240620

# 使用 Gemini 1.5 Pro
python3 run_client.py my_kernel.py --model-type gemini --model-id gemini-1.5-pro
```

### 常用可选参数
| 参数 | 说明 | 默认值 |
| :--- | :--- | :--- |
| `--model-type` | 模型提供商 (`openai`, `claude`, `gemini`) | `openai` |
| `--iters` | 编译重试与优化迭代的最大次数 | `3` |
| `--output-dir` | 结果输出目录 | `outputs` |
| `--target` | 目标硬件平台 | `MLU590` |
| `--quiet` | 静默模式，只输出最终结果 | `False` |

## 🔄 4. 自动化流程 (The Pipeline)

运行后，每个代码块会顺序经历以下四个阶段：

1.  **迁移 (Migration)**: 将 GPU 特有的语法（如 `tl.math`）转换为 MLU 语法，并进行初步的数据类型适配。
2.  **调试 (Auto-Debugging)**: Agent 自动生成临时文件并尝试编译。若报错，捕获 Error Log 反馈给 LLM 进行逻辑修复，直到代码“可运行”。
3.  **优化 (Optimization)**: 应用 **NRAM 对齐**、**双缓冲 (Double Buffering)**、**算子融合** 等策略压榨硬件性能。
4.  **调优 (Fine-tuning)**: 自动注入 `@triton.autotune`，针对 MLU 硬件特性寻找最优超参数。

## 📊 5. 结果展示与导出

任务完成后，Agent 会执行以下操作：
- **导出文件**: 在 `outputs/`（或你指定的目录）生成同名文件。
- **性能报告**: 在终端输出 ASCII 表格，展示各 Kernel 的编译情况、延迟 (Latency) 及加速比 (Speedup)。

## 🧪 6. 系统测试

在正式大规模使用前，建议运行以下测试确保组件正常：

```bash
# 测试 Prompt 模板生成逻辑
python3 -m unittest tests.test_templates

# 测试编排引擎与代码块替换逻辑
python3 -m unittest tests.test_orchestrator
```

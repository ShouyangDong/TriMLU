TriMLU: Automated Triton to MLU Migration Engine

# 1. Project Overview

**TriMLU** is an intelligent orchestration engine designed to automate the migration and optimization of OpenAI Triton kernels to Cambricon MLU hardware.

By leveraging Large Language Models (LLMs) and a specialized multi-stage pipeline, TriMLU bridges the gap between GPU-centric code and MLU-specific optimizations, significantly reducing the manual effort required for high-performance operator development.

# 2. Core Objectives

**Automated Migration**: Seamlessly translate CUDA-based Triton logic into MLU-optimized implementations.

**Iterative Optimization**: Use feedback loops (performance metrics and compiler errors) to refine kernels.

**Best-Practice Retrieval**: Utilize a curated corpus of MLU-optimized patterns to guide the generation process.

**Hardware-Awareness**: Target specific MLU architectures (e.g., MLU370) with tailored instructions and tiling strategies.

# 3. Current Architecture (V0.1)

The project is currently structured into several key modules:

- core/orchestrator.py: The "brain" that manages the migration stages (Parsing -> Migration -> Optimization -> Verification).

- prompts/selector.py: An intelligent retriever that finds relevant MLU code samples using semantic similarity.

- core/models/: Multi-provider support (OpenAI, Claude, Gemini) for code generation.

- core/tester.py: (Work in Progress) Integrated benchmarking and functional verification.

# 4. Future Roadmap

Phase 1: Foundation (Current - 2026.04)

- [x] Basic Orchestrator logic and pipeline flow.

- [x] Multi-LLM provider integration.

- [x] Example retrieval system based on Cosine Similarity.

- [x] Integration with Cambricon Triton toolchains for real-time compilation checks.

Phase 2: Intelligence & Accuracy (2026.05)

- [ ] Advanced Retrieval: Replace basic similarity with Vector Embeddings (e.g., ChromaDB or FAISS) for more precise logic matching.

- [ ] Chain of Thought (CoT): Implement specialized prompts that force the LLM to analyze MLU memory hierarchy before writing code.

- [ ] Error-Driven Refinement: Auto-parse Bang-C compiler errors and feed them back to the agent for self-correction.

Phase 3: Performance & Scale (2026.06)

- [x] Performance Profiling: Integrate cnperf data into the optimization loop to detect bank conflicts and instruction stalls.

- [ ] Multi-Kernel Support: Optimize complex models with interconnected kernels, ensuring efficient data movement between operators.

- [ ] Web UI/Dashboard: A visual interface to track migration progress and compare performance before/after migration.

Phase 4: Community & Ecosystem (2026.12)

- [ ] Open Corpus: Allow the community to contribute "Golden Kernels" to the reference library.

- [ ] Plugin System: Support for additional hardware backends (NPU, TPU) using the same orchestration logic.

Created by the TriMLU Dev Team.

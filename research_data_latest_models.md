# Research Data: Latest Models & Hardware for Energy-Efficient Generative AI Inference

> All data sourced from official announcements, published benchmarks, and peer-reviewed papers (2024–2026).
> **Last updated:** March 2026

---

## Part 1: Latest Large Language Models (LLMs)

### Table 1.1 — New MoE & Efficient LLM Architecture Comparison

| Model                           | Total Params | Active Params/Token | Architecture                             | Context Length                    | Release Date |
| ------------------------------- | ------------ | ------------------- | ---------------------------------------- | --------------------------------- | ------------ |
| GLM-5 (Zhipu AI)                | 744B         | ~40B                | Transformer MoE (256 experts, ~8 active) | 205K tokens                       | Feb 2026     |
| MiniMax M2.5                    | 230B         | 10B                 | MoE                                      | —                                 | Feb 2026     |
| LFM2-24B-A2B (Liquid AI)        | 24B          | 2B                  | Hybrid (non-Transformer MoE)             | 128K tokens                       | 2025         |
| Qwen3-30B-A3B (Alibaba)         | 30.5B        | 3.3B                | Transformer MoE                          | 32K (extendable to 131K via YaRN) | Apr 2025     |
| DeepSeek V3                     | 685B         | 37B                 | Transformer MoE (DeepSeekMoE + MLA)      | 128K tokens                       | Dec 2024     |
| GPT-OSS-20B (Existing in paper) | 20.9B        | 3.6B                | Sparse MoE                               | —                                 | 2024         |
| LLaMA 3-70B (Existing in paper) | 70B          | 70B (dense)         | Dense Transformer                        | 128K tokens                       | 2024         |

**Source URLs:**

- GLM-5: [z.ai/blog/glm-5](https://z.ai/blog/glm-5) | [NVIDIA Blog on GLM-5](https://blogs.nvidia.com/blog/zhipu-glm5-nemo/)
- MiniMax M2.5: [minimax.io/news/minimax-m25](https://www.minimax.io/news/minimax-m25)
- LFM2-24B-A2B: [huggingface.co/LiquidAI/LFM2-24B-A2B](https://huggingface.co/LiquidAI/LFM2-24B-A2B) | [liquid.ai/blog/lfm2-24b-a2b](https://www.liquid.ai/blog/lfm2-24b-a2b)
- Qwen3: [qwenlm.github.io/blog/qwen3](https://qwenlm.github.io/blog/qwen3/)
- DeepSeek V3: [arxiv.org/abs/2412.19437](https://arxiv.org/abs/2412.19437)

---

### Table 1.2 — LLM Inference Performance & Efficiency

| Model                      | Hardware                     | Throughput (tok/s)                       | Sparsity Ratio            | Key Efficiency Feature                                                   |
| -------------------------- | ---------------------------- | ---------------------------------------- | ------------------------- | ------------------------------------------------------------------------ |
| **GLM-5**                  | NVIDIA Blackwell/Hopper GPUs | Comparable to 30–70B dense model latency | 5.9% active (40B/744B)    | DeepSeek Sparse Attention (DSA) + MLA; reduces compute by ~95% per token |
| **MiniMax M2.5-Lightning** | API (cloud)                  | 100 tok/s                                | 4.3% active (10B/230B)    | Extreme sparsity with high throughput                                    |
| **MiniMax M2.5 Standard**  | API (cloud)                  | 50 tok/s                                 | 4.3% active               | Cost-optimized variant                                                   |
| **LFM2-24B-A2B**           | AMD CPU (Ryzen AI Max+ 395)  | 112 tok/s (decode)                       | 8.3% active (2B/24B)      | Hybrid architecture; runs in 32GB RAM on consumer devices                |
| **LFM2-24B-A2B**           | NVIDIA H100 (single GPU)     | 293 tok/s (decode)                       | 8.3% active               | Best-in-class on-device efficiency                                       |
| **Qwen3-30B-A3B**          | RTX 3090 (Q6 quant)          | ~73 tok/s                                | 10.8% active (3.3B/30.5B) | MoE with excellent consumer GPU performance                              |
| **Qwen3-30B-A3B**          | Apple M4 Max (Q4 quant, MLX) | >100 tok/s                               | 10.8% active              | Efficient on Apple Silicon                                               |
| **DeepSeek V3**            | H800 GPUs                    | —                                        | 5.4% active (37B/685B)    | MLA + aux-loss-free load balancing; FP8 mixed precision training         |
| **GPT-OSS-20B**            | A100 80GB                    | ~32 tok/s                                | 17.2% active (3.6B/20.9B) | ~4–5× more energy efficient than LLaMA-3.1-70B                           |

**Source URLs:**

- LFM2 throughput: [huggingface.co/LiquidAI/LFM2-24B-A2B](https://huggingface.co/LiquidAI/LFM2-24B-A2B)
- Qwen3 inference speeds: [reddit.com/r/LocalLLaMA (community benchmarks)](https://www.reddit.com/r/LocalLLaMA/)
- DeepSeek V3: [arxiv.org/abs/2412.19437](https://arxiv.org/abs/2412.19437)

---

### Key Findings — LLM Efficiency (New Data)

1. **Extreme sparsity is the dominant trend.** The latest MoE models activate only 4–11% of total parameters per token, compared to 17% for the earlier GPT-OSS-20B. This implies even greater energy savings than the 4–5× advantage documented in the paper.

2. **GLM-5's 744B model with only 40B active params** extends the paper's finding that MoE architecture dominates quantization for energy efficiency. With ~5.9% activation ratio and DSA attention, GLM-5 should theoretically consume energy comparable to a 40B dense model — far less than its 744B parameter count suggests.

3. **LFM2-24B-A2B's hybrid non-Transformer architecture** is a new paradigm. By achieving 112 tok/s on CPU and fitting in 32GB RAM with only 2B active parameters, this challenges the assumption that large models require GPU-class hardware. This has major energy implications for edge inference.

4. **Consumer hardware feasibility.** Qwen3-30B-A3B achieves >100 tok/s on Apple M4 Max and ~73 tok/s on RTX 3090, demonstrating that MoE models bring powerful inference to consumer devices without the energy overhead of server-class GPUs.

---

## Part 2: Latest Image Generation Models

### Table 2.1 — Image Generation Model Comparison

| Model                                      | Parameters           | Architecture                      | Inference Steps       | Max Resolution  | Key Performance                                                                   |
| ------------------------------------------ | -------------------- | --------------------------------- | --------------------- | --------------- | --------------------------------------------------------------------------------- |
| **Flux 1.1 Pro Ultra** (Black Forest Labs) | ~12B                 | Vision Transformer (DiT)          | 28 (Dev), 4 (Schnell) | 4MP (2048×2048) | ~10s per image; 2.5× faster than comparable models; #1 Elo on Artificial Analysis |
| **Z-Image** (Alibaba/Tongyi MAI)           | 6B                   | S3-DiT (Single-Stream DiT)        | 28–50                 | 4MP (2048×2048) | Photorealistic + bilingual text rendering (EN/CN)                                 |
| **Z-Image-Turbo**                          | 6B                   | S3-DiT (distilled)                | 8 (configurable to 1) | 4MP             | Sub-second on H800; fits in 16GB VRAM; #1 open-source on Artificial Analysis      |
| **GLM-Image** (Zhipu AI)                   | 7B diffusion + 9B AR | Single-stream DiT + AR (GLM-4-9B) | —                     | 1024–2048px     | Autoregressive + diffusion hybrid                                                 |
| SD v1.5 (In paper)                         | ~860M                | U-Net                             | 50                    | 512×512         | 0.15 Wh baseline                                                                  |
| SD-XL (In paper)                           | ~2.6B                | U-Net (large)                     | 50                    | 1024×1024       | 0.45 Wh baseline                                                                  |
| Flux Dev (In paper)                        | ~12B                 | Transformer                       | 28                    | —               | 0.85 Wh baseline                                                                  |
| Flux Schnell (In paper)                    | ~12B                 | Transformer                       | 4                     | —               | 0.52 Wh baseline                                                                  |

**Source URLs:**

- Flux 1.1 Pro Ultra: [bfl.ai](https://bfl.ai/) | [fal.ai/models/flux-pro-ultra](https://fal.ai/models/flux-pro/v1.1-ultra)
- Z-Image: [github.com/TongYiMAI/Z-Image](https://github.com/TongYiMAI/Z-Image) | [huggingface.co/TongYiMAI](https://huggingface.co/TongYiMAI)
- Z-Image-Turbo: [fal.ai/models/z-image-turbo](https://fal.ai/models/z-image/turbo) | [digitalocean.com benchmarks](https://www.digitalocean.com/)
- GLM-Image: [z.ai/blog/glm-image](https://z.ai/blog/glm-image)

---

### Key Findings — Image Generation (New Data)

1. **Step count reduction remains king.** Z-Image-Turbo with 8 steps (vs 28–50 for base models) confirms the paper's finding that reducing inference steps yields larger savings than quantization. This is now being achieved through distillation rather than just fast samplers.

2. **Single-Stream DiT (S3-DiT) as new architecture.** Z-Image's unified text+image token processing in a single stream reduces parameter overhead and computational redundancy, potentially addressing the higher energy cost of Transformer-based diffusion models noted in the paper.

3. **Consumer VRAM targets.** Z-Image-Turbo fitting in 16GB VRAM is significant for energy efficiency on consumer hardware, removing the need for multiple GPUs or high-power server cards.

4. **Autoregressive + diffusion hybrids.** GLM-Image combines a 9B autoregressive model with a 7B diffusion decoder, representing a new architectural pattern that may have different energy profiles than pure diffusion models.

---

## Part 3: Hardware Optimization & AI Accelerator Results

### Table 3.1 — AI Accelerator Comparison for Inference

| Accelerator                       | Vendor   | Peak FLOPS (FP8)          | HBM Capacity        | HBM Bandwidth | TDP (Watts)      | Key Inference Performance                                                |
| --------------------------------- | -------- | ------------------------- | ------------------- | ------------- | ---------------- | ------------------------------------------------------------------------ |
| **Ironwood (TPU v7)**             | Google   | 4,614 TFLOPS              | 192 GB              | 7.4 TB/s      | —                | 4× faster than TPU v6e; 2× perf/watt vs TPUv6e; 29.3 TFLOPS/W            |
| **B200 (Blackwell)**              | NVIDIA   | ~9,000 TFLOPS (FP4)       | 192 GB HBM3e        | 8 TB/s        | 1,000W           | 60K tok/s/GPU on gpt-oss; 2.3× H100 at FP16                              |
| **GB200 NVL72**                   | NVIDIA   | — (72-GPU system)         | Up to 13.8 TB       | —             | ~1,200W/GPU      | 869K tok/s on Llama 2 70B; 30× vs H100                                   |
| **GB300 NVL72 (Blackwell Ultra)** | NVIDIA   | 1.5× B200 FP4             | 288 GB HBM3e/GPU    | —             | ~1,400W/GPU      | 50× throughput/MW vs Hopper; 35× lower cost/token                        |
| **CS-3 / WSE-3**                  | Cerebras | 125 PFLOPS (est.)         | 44 GB SRAM (on-die) | —             | 23,000W (system) | 2,100 tok/s on Llama 3.2 70B (single user); 970 tok/s on Llama 3.1 405B  |
| **MI300X**                        | AMD      | 2,610 TFLOPS              | 192 GB HBM3         | 5.3 TB/s      | 750W             | 3,062 tok/s offline on Llama 2 70B (single GPU, MLPerf)                  |
| **MI325X**                        | AMD      | 2,600 TFLOPS              | 256 GB HBM3E        | 6.0 TB/s      | 1,000W           | 20–40% faster inference than H200 on Llama 3.1 70B; 1.31 TFLOPS/W (FP16) |
| **Colossus (xAI)**                | xAI      | System-level (200K H100s) | —                   | —             | 250 MW (system)  | Inference-optimized cluster; custom 500W Blackwell chips for inference   |
| **A100 80GB** (In paper)          | NVIDIA   | 624 TFLOPS                | 80 GB HBM2e         | 2 TB/s        | 400W             | Baseline in paper                                                        |
| **H100 SXM** (In paper)           | NVIDIA   | 3,958 TFLOPS              | 80 GB HBM3          | 3.35 TB/s     | 700W             | Baseline comparison                                                      |

**Source URLs:**

- Google Ironwood: [cloud.google.com/blog/products/ai-machine-learning/ironwood-tpu](https://cloud.google.com/blog/products/compute/ironwood-is-googles-7th-gen-tpu) | [blog.google/technology/google-deepmind/ironwood](https://blog.google/technology/google-deepmind/ironwood/)
- NVIDIA Blackwell B200: [nvidia.com/en-us/data-center/b200](https://www.nvidia.com/en-us/data-center/b200/) | [MLPerf Inference v5.0 results](https://mlcommons.org/benchmarks/inference-datacenter/)
- NVIDIA GB300 Ultra: [nvidia.com/en-us/data-center/gb300-nvl72](https://www.nvidia.com/en-us/data-center/gb300-nvl72/)
- Cerebras CS-3: [cerebras.ai/product-system](https://cerebras.ai/product-system/) | [cerebras.ai/inference](https://cerebras.ai/inference/)
- AMD MI300X: [amd.com/en/products/accelerators/instinct/mi300x](https://www.amd.com/en/products/accelerators/instinct/mi300x.html)
- AMD MI325X: [amd.com/en/products/accelerators/instinct/mi325x](https://www.amd.com/en/products/accelerators/instinct/mi325x.html)
- xAI Colossus: [rdworldonline.com](https://www.rdworldonline.com/) | [Wikipedia – Colossus](<https://en.wikipedia.org/wiki/Colossus_(supercomputer)>)

---

### Table 3.2 — Energy Efficiency Comparison Across Hardware Generations

| Accelerator               | Energy Efficiency vs Predecessor                 | Inference Perf/Watt Metric       | Key Innovation                                                              |
| ------------------------- | ------------------------------------------------ | -------------------------------- | --------------------------------------------------------------------------- |
| **Ironwood TPU v7**       | 2× perf/watt vs TPU v6e                          | 29.3 TFLOPS/W                    | 30× more efficient than 1st-gen Cloud TPU; ICI at 9.6 Tbps                  |
| **NVIDIA B200**           | 25× better per-token efficiency vs Hopper (H100) | 10× throughput/MW for MoE models | FP4 precision (new); 2nd-gen Transformer Engine                             |
| **NVIDIA GB300 NVL72**    | 50× throughput/MW vs Hopper                      | 35× lower cost per token         | Liquid cooling; 288GB HBM3e; 1.5× B200 performance                          |
| **Cerebras CS-3**         | 2× perf at same power vs CS-2                    | ~91 tok/s per kW (Llama 70B)     | Wafer-scale (4 trillion transistors); eliminates off-chip memory bottleneck |
| **AMD MI325X**            | ~1.3× vs MI300X                                  | 1.31 TFLOPS/W (FP16)             | 256GB HBM3E enables single-GPU large model inference                        |
| **AMD MI355X (upcoming)** | ~3× tokens/sec/MW vs MI300X                      | (Projected, mid-2025)            | CDNA4 architecture                                                          |

**Source URLs:**

- NVIDIA efficiency claims: [nvidia.com/en-us/data-center/solutions/inferencing](https://www.nvidia.com/en-us/data-center/solutions/inferencing/) | [SemiAnalysis InferenceMAX](https://www.semianalysis.com/)
- Google efficiency: [blog.google/technology/google-deepmind/ironwood](https://blog.google/technology/google-deepmind/ironwood/)
- Cerebras: [cerebras.ai](https://cerebras.ai/) | [Forbes](https://www.forbes.com/)
- AMD roadmap: [amd.com/en/products/accelerators/instinct](https://www.amd.com/en/products/accelerators/instinct.html)

---

### Key Findings — Hardware (New Data)

1. **Generational efficiency leaps.** NVIDIA Blackwell achieves 25× per-token efficiency vs Hopper, while GB300 Ultra pushes this to 50× throughput/MW. This fundamentally changes the energy landscape since the paper's A100/H100 measurements.

2. **New precision formats.** FP4 (NVIDIA Blackwell) and FP6 are emerging as viable inference precisions, offering 2× throughput over FP8. This extends the paper's quantization analysis beyond INT8/INT4.

3. **Wafer-scale computing (Cerebras).** The CS-3 achieving 2,100 tok/s on Llama 70B at 23kW system power (~91 tok/s/kW) represents a fundamentally different efficiency paradigm — eliminating off-chip memory access entirely.

4. **Custom inference silicon.** xAI's custom 500W Blackwell-derived inference chips (vs 700–1000W for general-purpose GPUs) show the trend toward inference-specialized, lower-power accelerators.

5. **Non-NVIDIA competition.** Google Ironwood's 2× perf/watt over TPU v6e and AMD MI325X's competitive inference performance at 1000W provide important reference points for cross-hardware energy analysis.

---

## Part 4: Cross-Cutting Analysis for Paper Integration

### How These Findings Extend the Paper's Results

| Paper Finding                                              | New Supporting/Extending Data                                                                                                           |
| ---------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| MoE 4–5× energy advantage over dense                       | GLM-5 (5.9% active), MiniMax M2.5 (4.3% active) push sparsity further, suggesting **10–20× theoretical savings** over dense equivalents |
| Quantization paradox for diffusion models                  | Z-Image-Turbo's distillation approach (8 steps) bypasses quantization entirely for speed gains; new paradigm vs INT8                    |
| DVFS yields 30–50% improvement                             | NVIDIA Blackwell's 25× per-token efficiency makes hardware generation choice more impactful than DVFS tuning alone                      |
| U-Net more efficient than Transformer diffusion            | S3-DiT (Z-Image) merges text+image streams, potentially reducing gap; Flux still uses more energy but with superior quality             |
| Step reduction > quantization for diffusion energy savings | Z-Image-Turbo (8 steps) and Flux Schnell (4 steps) confirm this; distillation is now the primary step-reduction technique               |

### Suggested New Tables for Paper

1. **Extended LLM Comparison (Table 1 update):** Add GLM-5, MiniMax M2.5, LFM2-24B-A2B, Qwen3-30B-A3B, DeepSeek V3 with their active parameter ratios
2. **Extended Diffusion Model Comparison (Table 3 update):** Add Z-Image, Z-Image-Turbo, Flux 1.1 Pro Ultra, GLM-Image
3. **New Table: AI Accelerator Energy Efficiency:** Compare Ironwood, Blackwell B200/GB300, Cerebras CS-3, AMD MI300X/MI325X
4. **New Table: Hardware Generation Impact:** Show how A100 → H100 → B200 → GB300 progression affects per-token energy

---

## Part 5: Additional References for the Paper

### New Citations to Add

| Ref # | Citation                                                                                                             | Topic                                    |
| ----- | -------------------------------------------------------------------------------------------------------------------- | ---------------------------------------- |
| [14]  | Zhipu AI, "GLM-5 Technical Report," z.ai/blog/glm-5, Feb 2026.                                                       | GLM-5 MoE architecture, 744B/40B active  |
| [15]  | MiniMax, "MiniMax M2.5: Built for Real-World Productivity," minimax.io/news/minimax-m25, Feb 2026.                   | MiniMax M2.5, 230B/10B MoE               |
| [16]  | Liquid AI, "LFM2-24B-A2B Model Card," huggingface.co/LiquidAI/LFM2-24B-A2B, 2025.                                    | Hybrid 24B/2B active architecture        |
| [17]  | Alibaba Qwen Team, "Qwen3 Technical Report," qwenlm.github.io/blog/qwen3, Apr 2025.                                  | Qwen3-30B-A3B MoE                        |
| [18]  | DeepSeek-AI, "DeepSeek-V3 Technical Report," arXiv:2412.19437, Dec 2024.                                             | DeepSeek V3 685B MoE, MLA, FP8           |
| [19]  | Alibaba Tongyi MAI Lab, "Z-Image: Scalable Single-Stream Diffusion Transformer," github.com/TongYiMAI/Z-Image, 2025. | Z-Image/Turbo, S3-DiT architecture       |
| [20]  | Black Forest Labs, "Flux 1.1 Pro Ultra," bfl.ai, Nov 2024.                                                           | Flux 1.1 Pro Ultra, 4MP generation       |
| [21]  | Google, "Ironwood: Our 7th Generation TPU," cloud.google.com/blog, Apr 2025.                                         | TPU v7 specs, 4614 TFLOPS FP8            |
| [22]  | NVIDIA, "Blackwell Architecture Technical Brief," nvidia.com, 2024.                                                  | B200, GB200 specs and efficiency         |
| [23]  | NVIDIA, "GB300 NVL72 Product Brief," nvidia.com, 2025.                                                               | 50× throughput/MW vs Hopper              |
| [24]  | Cerebras Systems, "CS-3 and WSE-3," cerebras.ai, 2024.                                                               | Wafer-scale inference, 2100 tok/s on 70B |
| [25]  | AMD, "Instinct MI325X Accelerator," amd.com, Oct 2024.                                                               | 256GB HBM3E, inference performance       |
| [26]  | NVIDIA, "MLPerf Inference v5.0 Results," nvidia.com/en-us/data-center, Apr 2025.                                     | GB200 NVL72 benchmark results            |
| [27]  | TokenPowerBench, "Benchmarking LLM Inference Energy," ResearchGate/arXiv, Dec 2025.                                  | Power consumption per token benchmarks   |
| [28]  | ML.ENERGY, "Leaderboard v3.0," ml.energy, Dec 2025.                                                                  | Joules per token across models           |

---

## Part 6: Energy Benchmarking Resources

| Resource              | URL                                                                                                                   | What It Provides                                        |
| --------------------- | --------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------- |
| ML.ENERGY Leaderboard | [ml.energy](https://ml.energy/)                                                                                       | Joules/token for various LLMs on different GPUs         |
| TokenPowerBench       | [arXiv (Dec 2025)](https://arxiv.org/)                                                                                | Systematic power measurement across models/hardware     |
| Artificial Analysis   | [artificialanalysis.ai](https://artificialanalysis.ai/)                                                               | Token throughput, latency, and pricing across providers |
| MLPerf Inference      | [mlcommons.org/benchmarks](https://mlcommons.org/benchmarks/inference-datacenter/)                                    | Standardized inference benchmarks including power       |
| Hugging Face Open LLM | [huggingface.co/spaces/open-llm-leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) | Quality metrics for comparison                          |

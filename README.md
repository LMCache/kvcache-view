# KV Cache Growth Visualization

> 🎯 **Watch memory explode as LLMs scale to millions of tokens!**

An impactful, animated visualization demonstrating the exponential memory growth challenge of KV caches in Large Language Models, inspired by [LMCache's](https://lmcache.ai) memory optimization research.

## ✨ Features

- 🚀 **Real-time Animation**: Watch KV cache grow from 0 to 100M tokens
- 🔄 **Model Comparison**: Switch between 20+ models including Llama, Qwen3-Next, Qwen3-Omni, Gemma, Mixtral MOE, and DeepSeek
- 📊 **SOTA Context Lengths**: 128K, 200K, 1M, 2M, 10M, up to 100M tokens
- 🎚️ **Dynamic Data Types**: FP32, FP16, BF16, INT8, INT4 quantization
- ⚡ **Speed Control**: 0.5x to 100x animation speed
- 📈 **Live Metrics**: Memory usage, GPU requirements, efficiency calculations
- 🏋️ **Training Visualization**: Interactive training memory visualization with weights, gradients, optimizer states
- 🖥️ **Multi-GPU Support**: Distributed inference simulation with interconnect bandwidth modeling
- 🎮 **Latest GPUs**: Support for NVIDIA Blackwell (B100, B200, GB200) and AMD MI300X
- 🌊 **Beautiful Visuals**: Dynamic particle animations and visual effects
- 📱 **Progressive Web App**: Install as a full-screen standalone app on mobile and desktop devices

## 🚀 Live Demo

### View Online (No Installation Required!)
**[🖥️ Inference Visualization](https://htmlpreview.github.io/?https://github.com/LMCache/kvcache-view/blob/main/inference.html)** - KV Cache growth simulation

**[🏋️ Training Visualization](https://htmlpreview.github.io/?https://github.com/LMCache/kvcache-view/blob/main/train.html)** - Training memory explosion

Or visit: **[kvcache.io](https://kvcache.io/)**

<div align="center">
  <img src="kvcache-qr.png" alt="QR Code for kvcache.io" width="200"/>
  <p><em>Scan to visit kvcache.io</em></p>
</div>

Direct URLs:
```
# Inference (KV Cache) Visualization
https://htmlpreview.github.io/?https://github.com/LMCache/kvcache-view/blob/main/inference.html

# Training Memory Visualization
https://htmlpreview.github.io/?https://github.com/LMCache/kvcache-view/blob/main/train.html
```

**Note:** GitHub's HTML preview may have slight performance limitations compared to running locally.

### 📱 Install as Progressive Web App (PWA)

The visualization supports installation as a standalone app on your device:

**On Mobile (Android/iOS):**
1. Visit [kvcache.io](https://kvcache.io/) in Chrome (Android) or Safari (iOS)
2. Tap the browser menu (⋮ or share icon)
3. Select "Add to Home Screen" or "Install App"
4. The app will launch in full-screen mode without browser UI

**On Desktop (Chrome/Edge):**
1. Visit [kvcache.io](https://kvcache.io/)
2. Click the install icon (⊕) in the address bar
3. Or go to Menu → "Install kvcache.io..."
4. Launch from your applications menu or desktop

**PWA Benefits:**
- ✅ Full-screen experience without browser chrome
- ✅ Works offline after initial load
- ✅ Faster loading with service worker caching
- ✅ App-like interface on mobile devices
- ✅ Add to home screen with custom icon

**Verify PWA Support:**
```bash
./check-pwa.sh https://kvcache.io/
```

The included `check-pwa.sh` script validates all PWA requirements (manifest, service worker, icons, meta tags).

## The Memory Wall Problem

As context lengths grow from thousands to millions of tokens, KV cache memory becomes the critical bottleneck. This visualization makes that challenge tangible and visceral.

## What It Shows

- **Real-time Memory Growth**: Watch as KV cache memory explodes with increasing context length
- **Model Comparisons**: Switch between 20+ models from 1B to 671B parameters including MOE architectures
- **GPU Requirements**: Live calculation of modern GPUs needed as memory exceeds single-GPU capacity
- **Training Memory**: Visualize weights, gradients, optimizer states, and activations during training
- **Multi-GPU Scaling**: See how distributed inference works across multiple GPUs with interconnect modeling
- **Visual Metaphors**:
  - Memory grid fills up showing utilization
  - Exponential curve traces the non-linear growth
  - Particle effects intensify with memory pressure
  - Dynamic visual patterns representing memory growth

## Key Insights Visualized

1. **Small Models (1-4B)**: Can handle 1M context on single datacenter GPUs
2. **Medium Models (8-32B)**: Quickly exceed single GPU at moderate context lengths
3. **MOE Models (Mixtral, Phi-3.5 MoE)**: Efficient inference with sparse activation
4. **Large Models (70-405B)**: Require 6+ H100s for 1M context
5. **DeepSeek-V3 (671B)**: 7x memory reduction through KV-LoRA compression
6. **Qwen3-Next-80B**: Revolutionary hybrid architecture combining Gated DeltaNet (linear attention) with traditional attention in a 3:1 ratio, reducing KV cache by 75% while using only 3B of 80B parameters per token ([arXiv:2505.09388](https://arxiv.org/abs/2505.09388) [Audio summary](https://open.spotify.com/episode/0uS9uYJOcSELtibF7LfWuU))
7. **Qwen3-Omni (30B)**: Multimodal model optimized for text, audio, and vision with MoE architecture ([arXiv:2509.17765](https://arxiv.org/abs/2509.17765) [Audio summary](https://open.spotify.com/episode/0uS9uYJOcSELtibF7LfWuU))

## 🎯 Quick Start

### Option 1: Use Make (Recommended)
```bash
make        # Starts server and opens browser automatically
make stop   # Stop the server
```

### Option 2: Python HTTP Server
```bash
python3 -m http.server 8000
# Then navigate to http://localhost:8000
```

### Option 3: Open Directly
```bash
open index.html      # macOS
xdg-open index.html  # Linux
start index.html     # Windows
```

### Option 4: GitHub Pages
Fork this repo and enable GitHub Pages in settings, then visit:
```
https://[your-username].github.io/kvcache-view/
```

## Controls

- **Play/Pause**: Start or stop the animation
- **Speed Control**: Adjust animation speed (0.5x to 100x) - now with ludicrous speed!
- **Next Model**: Cycle through different model architectures
- **Context Control**: Switch between SOTA context lengths:
  - 128K - Standard context
  - 200K - Claude 3.5 Sonnet
  - 1M - Llama 3.1 / standard long context
  - 2M - Gemini 1.5 Pro
  - 10M - Magic/research models
  - 100M - Theoretical future (prepare for memory apocalypse!)
- **Type Control**: Dynamic data type switching:
  - FP32 - Full precision (4 bytes)
  - FP16 - Half precision (2 bytes)
  - BF16 - Brain float (2 bytes)
  - INT8 - 8-bit quantization (1 byte)
  - INT4 - 4-bit quantization (0.5 bytes)

## The Math Behind It

The visualization uses the exact KV cache formulas from [LMCache's KV Cache Calculator](https://lmcache.ai/kv_cache_calculator.html):

### Standard Models
```
KV Cache = 2 × layers × tokens × kv_heads × (hidden_size/attention_heads) × dtype_size
```

### DeepSeek Models (with KV-LoRA)
```
KV Cache = layers × tokens × (kv_lora_rank + qk_rope_head_dim) × dtype_size
```

## Model Architecture Innovations

### Qwen3-Next-80B: Breaking the Memory Wall
Based on the [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388) ([Audio summary](https://open.spotify.com/episode/0uS9uYJOcSELtibF7LfWuU)), Qwen3-Next-80B introduces several groundbreaking optimizations:

- **Hybrid Attention Architecture**: 75% Gated DeltaNet (linear complexity) + 25% traditional attention for optimal memory-performance trade-off
- **Ultra-Sparse MoE**: 512 experts with only 11 activated per token, achieving 80B total capacity with 3B active parameters
- **Multi-Token Prediction**: Accelerates inference and improves speculative decoding acceptance rates
- **Long Context Optimization**: Native 262K context support, expandable to 1M tokens with minimal memory overhead

**Performance Gains**: 7-10x faster prefill, 4-10x faster decode, while using <80% of traditional model GPU hours.

### DeepSeek-V3: KV-LoRA Innovation
Uses KV-LoRA compression to achieve 7x memory reduction compared to standard transformer architectures.

## Why These Optimizations Matter

LMCache and modern model architectures address memory explosion through:
- Prefix caching to reuse computed KV states
- Compression techniques (KV-LoRA, quantization)
- Distributed caching across multiple nodes
- Architectural innovations (hybrid attention, sparse MoE)

This visualization demonstrates why such optimizations are critical for the future of LLMs.

## Inspiration

- **Technical Foundation**: Based on LMCache's KV cache calculator and memory projections
- **Purpose**: Make the abstract concept of memory scaling tangible and impactful through interactive visualization

## Technical Stack

- Pure JavaScript with HTML5 Canvas
- No external dependencies
- Hardware-accelerated rendering
- Responsive design

## Memory Reference Tables

### Standard FP16 Precision

| Model | 128K Context | 1M Context | 2M Context | 10M Context | H100s @10M |
|-------|-------------|------------|-------------|-------------|------------|
| Llama-3.2-1B | 4 GiB | 31 GiB | 62 GiB | 312 GiB | 4 |
| Llama-3.1-8B | 16 GiB | 122 GiB | 244 GiB | 1.2 TiB | 16 |
| Llama-3.1-70B | 40 GiB | 305 GiB | 610 GiB | 3.0 TiB | 39 |
| DeepSeek-V3 | 8.5 GiB | 65 GiB | 131 GiB | 654 GiB | 9 |
| Llama-3.1-405B | 63 GiB | 481 GiB | 962 GiB | 4.7 TiB | 61 |

### Impact of Data Types (Llama-70B @ 1M tokens)

| Data Type | Memory Size | H100 GPUs | Memory Reduction |
|-----------|-------------|-----------|------------------|
| FP32 | 610 GiB | 8 | Baseline |
| FP16 | 305 GiB | 4 | 50% |
| BF16 | 305 GiB | 4 | 50% |
| INT8 | 153 GiB | 2 | 75% |
| INT4 | 76 GiB | 1 | 87.5% |

## 📁 Project Structure

```
kvcache-view/
├── index.html                    # Visualizations hub (entry page)
├── inference.html                # Inference visualization (KV cache growth)
├── visualization.js              # Inference visualization logic
├── train.html                    # Training visualization (memory explosion)
├── train-visualization.js        # Training visualization logic
├── manifest.json                 # PWA manifest (app metadata)
├── sw.js                         # Service worker (offline support)
├── icon-192.png                  # PWA icon (192x192)
├── icon-512.png                  # PWA icon (512x512)
├── icon.svg                      # Source SVG icon
├── check-pwa.sh                  # PWA support verification script
├── README.md                     # This file
├── Makefile                      # Simple server commands
├── CLAUDE.md                     # Development guidelines
├── GPU_OPTIMIZATION_DEFAULTS.md # GPU optimization rationale
└── VISUALIZATION_GUIDE.md       # Visual components guide
```

## 📚 Documentation

- **[Complete Feature Documentation](SIMULATION_FEATURES.md)** - Comprehensive guide to all simulation features, controls, and calculations
- **[GPU Optimization Defaults](GPU_OPTIMIZATION_DEFAULTS.md)** - Why certain optimizations (CB, PA, FA) are enabled/disabled by default for different GPUs
- **[Visualization Guide](VISUALIZATION_GUIDE.md)** - Detailed explanation of all visual components and their meanings
- **[Development Guidelines](CLAUDE.md)** - Contributing guidelines and common pitfalls

## 🛠️ Developer Tools

### PWA Support Checker

The `check-pwa.sh` script verifies Progressive Web App support on any website:

```bash
# Check kvcache.io (default)
./check-pwa.sh

# Check any other website
./check-pwa.sh https://example.com
```

**Checks performed:**
- ✅ Web app manifest (manifest.json)
- ✅ Manifest accessibility and content validation
- ✅ Theme color meta tag
- ✅ Apple mobile web app support
- ✅ Service worker registration
- ✅ App icons (standard and Apple touch icons)
- ✅ Viewport meta tag

**Example output:**
```
=== PWA Support Checker ===
Checking: https://kvcache.io/

1. Manifest link: ✓ Found (manifest.json)
   Manifest accessible: ✓ Yes
   Preview: KV Cache Memory Visualization - Interactive visualization...
2. Theme color: ✓ Found (#1428a0)
3. Apple mobile web app: ✓ Found
4. Service worker: ✓ Found (./sw.js)
5. App icons: ✓ Found (2 standard, 1 Apple)
6. Viewport meta: ✓ Found

=== Summary ===
PWA Support: 6/6 checks passed (100%)
✓ Good PWA support detected!
```

The script provides color-coded output with pass/fail indicators and a percentage score.

## 🔗 Related Projects

- [LMCache](https://github.com/LMCache/LMCache) - The KV cache optimization framework that inspired this visualization
- [LMCache Calculator](https://lmcache.ai/kv_cache_calculator.html) - Interactive KV cache size calculator

## 📝 License

Created as a demonstration of the memory challenges that LMCache solves. Open source under permissive licensing.

# llama-server-select

A PowerShell-based model selector and launcher for [llama.cpp](https://github.com/ggml-org/llama.cpp) server, optimized for running various Large Language Models (LLMs) on systems with limited VRAM (e.g., 8GB RTX 3070) and high RAM (64GB+).

## Features

- **Model Selection Menu**: Interactive menu to choose from pre-configured models
- **Optimized Parameters**: Model-specific sampling parameters and performance flags for best results
- **VRAM Management**: MoE layer offloading to CPU for VRAM-constrained GPUs
- **KV Cache Quantization**: Efficient memory usage for longer contexts
- **Logging**: Automatic logging of server output with timestamps
- **Port Management**: Automatic detection and stopping of conflicting servers

## Prerequisites

### Required Software
- **Windows 10/11** with PowerShell 5.1+
- **llama.cpp**: Built with server support
  - Clone: `git clone https://github.com/ggml-org/llama.cpp`
  - Build: Follow [build instructions](https://github.com/ggml-org/llama.cpp/blob/master/README.md)
  - Ensure `llama-server.exe` is in `.\build\bin\llama-server.exe`

### Required Models
Download the following GGUF model files to your `$LLM_DIR` (default: `H:\LLM`):

1. **Qwen3-Coder-30B**: [unsloth/Qwen3-Coder-30B-A3B-Instruct](https://huggingface.co/unsloth/Qwen3-Coder-30B-A3B-Instruct)
2. **Qwen3-30B Thinking**: [unsloth/Qwen3-30B-A3B-Thinking-2507-GGUF](https://huggingface.co/unsloth/Qwen3-30B-A3B-Thinking-2507-GGUF)
3. **Qwen3-VL-30B**: [unsloth/Qwen3-VL-30B-A3B-Instruct-GGUF](https://huggingface.co/unsloth/Qwen3-VL-30B-A3B-Instruct-GGUF) + mmproj file
4. **Nemotron-3-Nano**: [unsloth/Nemotron-3-Nano-30B-A3B](https://huggingface.co/unsloth/Nemotron-3-Nano-30B-A3B)
5. **GLM-4.6V-Flash**: [GLM-4.6V-Flash](https://huggingface.co/THUDM/glm-4-9b-chat) + mmproj file
6. **MedGemma-1.5-4B**: [unsloth/medgemma-1.5-4b-it](https://huggingface.co/unsloth/medgemma-1.5-4b-it) + mmproj file
7. **GPT-OSS-120B**: [openai/gpt-oss](https://github.com/openai/gpt-oss)
8. **LFM2.5-1.2B**: [LiquidAI/LFM2.5-1.2B-Instruct-GGUF](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF)

### Hardware Requirements
- **GPU**: NVIDIA GPU with CUDA support (tested on RTX 3070 8GB)
- **RAM**: 64GB+ recommended for MoE models
- **CPU**: Multi-core CPU for MoE expert offloading

## Installation

1. **Clone this repository**:
   ```bash
   git clone https://github.com/yourusername/llama-server-select.git
   cd llama-server-select
   ```

2. **Update paths in `llama.ps1`**:
   - `$SERVER_PATH`: Path to `llama-server.exe`
   - `$LLM_DIR`: Directory containing your model files
   - Model paths in `$MODELS` hashtable

3. **Set environment variable** (optional):
   ```powershell
   $env:LLM_API_KEY = "your-api-key-here"
   ```

4. **Run the script**:
   ```powershell
   .\llama.ps1
   ```

## Usage

1. **Launch the script**:
   ```powershell
   .\llama.ps1
   ```

2. **Select a model** from the menu (1-8)

3. **Server starts** with optimized parameters for the selected model

4. **Access the API** at `http://localhost:8080` (default port)

5. **Stop server**: Press 'S' in the menu or kill the process

### Model Categories

| Model | Category | Context | VRAM | Notes |
|-------|----------|---------|------|-------|
| Qwen3-Coder-30B | Coding Expert | 65K | ~7GB | Best for code generation |
| Qwen3-30B Thinking | Logic/Reasoning | 32K | ~7GB | Advanced reasoning |
| Qwen3-VL-30B | Vision | 65K | ~7GB | Multimodal (text + images) |
| Nemotron-3-Nano | Deep Thinking | 32K | ~7GB | NVIDIA's reasoning model |
| GLM-4.6V-Flash | Ultra Vision/Fast | 131K | ~6GB | Fast multimodal |
| MedGemma-1.5-4B | Medical | 16K | ~5GB | Medical expertise |
| GPT-OSS-120B | Heavy MoE | 32K | ~8GB | Massive MoE model |
| LFM2.5-1.2B | Liquid AI Edge | 32K | ~2GB | Lightweight edge model |

## Configuration

### Performance Tuning

The script includes optimized parameters for each model:

- **MoE Offloading**: Experts offloaded to CPU to save VRAM
- **KV Cache Quantization**: 4-bit quantization for memory efficiency
- **Flash Attention**: Enabled for better performance
- **Batch Processing**: Optimized batch sizes

### Customizing Models

Edit the `$MODELS` hashtable in `llama.ps1` to:
- Add new models
- Update file paths
- Modify parameters

### Environment Variables

- `LLM_API_KEY`: Optional API key for server authentication

## Troubleshooting

### Common Issues

1. **"llama-server not found"**
   - Ensure llama.cpp is built and `$SERVER_PATH` is correct

2. **"Model file not found"**
   - Check model paths in `$MODELS`
   - Verify files exist and are accessible

3. **Port already in use**
   - Script will prompt to stop existing server
   - Or manually kill process on port 8080

4. **Out of memory**
   - Reduce `-ngl` (GPU layers)
   - Increase MoE offloading
   - Use smaller models

### Logs

Server logs are saved in `logs/` directory with timestamps:
- Format: `{model_alias}_{timestamp}.log`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Test on your hardware
4. Submit a pull request

## License

MIT License - see LICENSE file for details

## Acknowledgments

- [llama.cpp](https://github.com/ggml-org/llama.cpp) for the server implementation
- Model providers: Unsloth, NVIDIA, Zhipu AI, Google, OpenAI, Liquid AI
- Community optimizations and best practices

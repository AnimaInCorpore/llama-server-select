# LLM Manager - RTX 3070 8GB / 64GB RAM
# ==========================================================
# PC SPEC: Ryzen 5900HX (8C/16T) | RTX 3070 (8GB VRAM) | 64GB DDR4 RAM
# SERVER REF: https://github.com/ggml-org/llama.cpp/discussions
# ==========================================================
#
# ==================== OPTIMIZATION REFERENCE ====================
#
# SAMPLING PARAMETERS BY MODEL TYPE:
# ----------------------------------
# | Model Type        | temp | top_p | top_k | min_p | repeat_penalty |
# |-------------------|------|-------|-------|-------|----------------|
# | Qwen3-Coder       | 0.7  | 0.8   | 20    | -     | 1.05           |
# | Qwen3-Thinking    | 0.6  | 0.95  | 20    | 0     | -              |
# | Qwen3-VL (Vision) | 0.1  | 0.95  | 20    | -     | -              |
# | Nemotron Reason   | 1.0  | 1.0   | -     | -     | -              |
# | Nemotron Tools    | 0.6  | 0.95  | -     | -     | -              |
# | GLM-4.6V-Flash    | 0.8  | 0.6   | 2     | 0.0   | 1.1            |
# | General/Default   | 0.7  | 0.95  | 40    | 0.05  | -              |
#
# MOE LAYER OFFLOADING (for 8GB VRAM + high RAM setups):
# ------------------------------------------------------
# -ot ".ffn_.*_exps.=CPU"           # Offload ALL MoE experts to CPU (most VRAM savings)
# -ot ".ffn_(up|down)_exps.=CPU"    # Offload up+down projections (moderate savings)
# -ot ".ffn_(up)_exps.=CPU"         # Offload only up projection (least savings)
#
# Custom layer offload example (layers 6+ only):
# -ot "\.(6|7|8|9|[0-9][0-9])\.ffn_(gate|up|down)_exps.=CPU"
#
# KV CACHE QUANTIZATION (for longer context):
# -------------------------------------------
# -ctk q4_0    # K cache 4-bit (default, good balance)
# -ctv q4_0    # V cache 4-bit (requires -fa on / flash attention)
# Options: f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1
#
# PERFORMANCE FLAGS:
# ------------------
# -fa on           # Flash attention (required for -ctv quantization)
# --mlock          # Lock model in RAM (prevents swapping)
# -ngl 99          # Offload all possible layers to GPU
# --n-cpu-moe N    # Number of CPU threads for MoE layers
# -b 512           # Batch size for prompt processing
# -ub 1024         # Micro-batch size (for small models like LFM)
# -np N            # Number of parallel sequences
#
# MODEL-SPECIFIC NOTES:
# ---------------------
# GLM-4.6V: May reason in Chinese. Use: --system-prompt "Respond in English and reason in English"
# Nemotron: Use enable_thinking=true for reasoning, false for faster non-reasoning responses
# Qwen3-Thinking: Supports 256K context, recommend >131K for complex reasoning
# Vision models: Require --mmproj for the vision encoder
#
# DOCUMENTATION SOURCES:
# ----------------------
# Qwen3-Coder:    https://huggingface.co/unsloth/Qwen3-Coder-30B-A3B-Instruct
# Qwen3-Thinking: https://huggingface.co/unsloth/Qwen3-30B-A3B-Thinking-2507-GGUF
# Qwen3-VL:       https://huggingface.co/unsloth/Qwen3-VL-30B-A3B-Instruct-GGUF
# Nemotron:       https://huggingface.co/unsloth/Nemotron-3-Nano-30B-A3B
# GLM-4.6:        https://unsloth.ai/docs/models/glm-4.6-how-to-run-locally
# LFM2.5:         https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF
# llama.cpp:      https://github.com/ggml-org/llama.cpp/blob/master/examples/server/README.md
#
# NOTE: When adding new models or updating existing ones, always check the
# model's HuggingFace page or official docs (URLs above) for the latest
# "Best Practices" or "Recommended Settings" section. Parameters may change
# between model versions!
#
# =================================================================

# --- CONFIGURATION ---
$SERVER_PATH = ".\build\bin\llama-server.exe"
$LLM_DIR = "H:\LLM"
$DEFAULT_PORT = 8080
$LOG_DIR = "$PSScriptRoot\logs"
# ---------------------

# Ensure log directory exists
if (-not (Test-Path $LOG_DIR)) {
    New-Item -ItemType Directory -Path $LOG_DIR -Force | Out-Null
}

# Validate server executable
if (-not (Test-Path $SERVER_PATH)) {
    Write-Host "ERROR: llama-server not found at: $SERVER_PATH" -ForegroundColor Red
    Write-Host "Please build llama.cpp first or update SERVER_PATH." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Optional: LLM_API_KEY (no longer required)
if (-not $env:LLM_API_KEY) {
    Write-Host "==========================================================" -ForegroundColor Yellow
    Write-Host "INFO: LLM_API_KEY not set - server runs without auth." -ForegroundColor Yellow
    Write-Host "==========================================================" -ForegroundColor Yellow
    Start-Sleep -Seconds 1
}

# Set window title (ignore if host doesn't support it)
try {
    $Host.UI.RawUI.WindowTitle = "LLM Manager - RTX 3070 8GB / 64GB RAM"
} catch { }

# Model definitions with metadata
$MODELS = @{
    '1' = @{
        Name = "Qwen3-Coder-30B"; Context = "65k"; Category = "Coding Expert"
        Path = "C:\Temp\Qwen3-Coder-30B-A3B-Instruct-Q8_0.gguf"
        VRAM = "~7GB"; MoE = $true
    }
    '2' = @{
        Name = "Qwen3-30B Thinking"; Context = "32k"; Category = "Logic/Reasoning"
        Path = "$LLM_DIR\Qwen3-30B-A3B-UD-Q8_K_XL.gguf"
        VRAM = "~7GB"; MoE = $true
    }
    '3' = @{
        Name = "Qwen3-VL-30B"; Context = "65k"; Category = "Vision"
        Path = "$LLM_DIR\Qwen3-VL-Instruct\Qwen3-VL-30B-A3B-Instruct-UD-Q8_K_XL.gguf"
        VRAM = "~7GB"; MoE = $true
    }
    '4' = @{
        Name = "Nemotron-3-Nano"; Context = "32k"; Category = "Deep Thinking"
        Path = "$LLM_DIR\Nemotron-3-Nano-30B-A3B-UD-Q8_K_XL.gguf"
        VRAM = "~7GB"; MoE = $true
    }
    '5' = @{
        Name = "GLM-4.6V-Flash"; Context = "131k"; Category = "Ultra Vision/Fast"
        Path = "$LLM_DIR\GLM-4.6V-Flash\GLM-4.6V-Flash-UD-Q4_K_XL.gguf"
        VRAM = "~6GB"; MoE = $true
    }
    '6' = @{
        Name = "MedGemma-1.5-4B"; Context = "16k"; Category = "Medical"
        Path = "$LLM_DIR\MedGemma-1.5\MedGemma-1.5-4b-it-UD-Q8_K_XL.gguf"
        VRAM = "~5GB"; MoE = $false
    }
    '7' = @{
        Name = "GPT-OSS-120B"; Context = "32k"; Category = "Heavy MoE"
        Path = "C:\Temp\gpt-oss-120b-mxfp4-00001-of-00003.gguf"
        VRAM = "~8GB"; MoE = $true
    }
    '8' = @{
        Name = "LFM2.5-1.2B"; Context = "32k"; Category = "Liquid AI Edge"
        Path = "$LLM_DIR\LFM2.5-1.2B-Instruct-BF16.gguf"
        VRAM = "~2GB"; MoE = $false
    }
}

function Get-FileSize([string]$Path) {
    if (Test-Path $Path) {
        $size = (Get-Item $Path).Length / 1GB
        return "{0:N1}GB" -f $size
    }
    return "N/A"
}

function Test-PortInUse([int]$Port) {
    $conn = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue
    return $null -ne $conn
}

function Stop-ServerOnPort([int]$Port) {
    $conn = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue
    if ($conn) {
        $proc = Get-Process -Id $conn.OwningProcess -ErrorAction SilentlyContinue
        if ($proc) {
            Write-Host "Stopping $($proc.ProcessName) (PID: $($proc.Id)) on port $Port..." -ForegroundColor Yellow
            Stop-Process -Id $proc.Id -Force
            Start-Sleep -Seconds 1
            return $true
        }
    }
    return $false
}

function Show-Menu {
    Clear-Host
    Write-Host "==========================================================" -ForegroundColor Cyan
    Write-Host "          AI THOUGHT PARTNER - MODEL SELECTOR (2026)" -ForegroundColor Green
    Write-Host "==========================================================" -ForegroundColor Cyan
    
    foreach ($key in ($MODELS.Keys | Sort-Object)) {
        $m = $MODELS[$key]
        $size = Get-FileSize $m.Path
        $status = if (Test-Path $m.Path) { "[OK]" } else { "[MISSING]" }
        $statusColor = if (Test-Path $m.Path) { "Green" } else { "Red" }
        
        Write-Host -NoNewline "$key) $($m.Name.PadRight(20))"
        Write-Host -NoNewline "[$($m.Context.PadRight(5))] " -ForegroundColor DarkCyan
        Write-Host -NoNewline "$($m.Category.PadRight(18))" -ForegroundColor Gray
        Write-Host -NoNewline "VRAM:$($m.VRAM.PadRight(6))" -ForegroundColor DarkYellow
        Write-Host -NoNewline "Size:$($size.PadRight(7))" -ForegroundColor DarkGray
        Write-Host $status -ForegroundColor $statusColor
    }
    
    Write-Host "----------------------------------------------------------"
    if (Test-PortInUse $DEFAULT_PORT) {
        Write-Host "S) Stop server on port $DEFAULT_PORT" -ForegroundColor Red
    }
    Write-Host "X) Exit"
    Write-Host "==========================================================" -ForegroundColor Cyan
}

function Start-LLMServer {
    param(
        [string]$ModelPath,
        [string]$Alias,
        [string[]]$Arguments
    )

    # Validate model file exists
    if (-not (Test-Path $ModelPath)) {
        Write-Host ""
        Write-Host "ERROR: Model file not found!" -ForegroundColor Red
        Write-Host "Path: $ModelPath" -ForegroundColor Red
        Write-Host ""
        Read-Host "Press Enter to continue"
        return
    }

    # Check if port is in use
    if (Test-PortInUse $DEFAULT_PORT) {
        Write-Host ""
        Write-Host "WARNING: Port $DEFAULT_PORT is already in use!" -ForegroundColor Yellow
        $response = Read-Host "Stop existing server? [Y/N]"
        if ($response -eq 'Y' -or $response -eq 'y') {
            Stop-ServerOnPort $DEFAULT_PORT
        } else {
            return
        }
    }

    # Append API key only if it exists
    if ($env:LLM_API_KEY) {
        $allArgs = @($Arguments + @('--api-key', $env:LLM_API_KEY))
    } else {
        $allArgs = $Arguments
    }

    # Generate log filename
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $logFile = "$LOG_DIR\${Alias}_$timestamp.log"

    Write-Host ""
    Write-Host "==========================================================" -ForegroundColor Green
    Write-Host "Starting: $Alias" -ForegroundColor Green
    Write-Host "Model: $ModelPath" -ForegroundColor DarkGray
    Write-Host "Log: $logFile" -ForegroundColor DarkGray
    Write-Host "==========================================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Server arguments:" -ForegroundColor Cyan
    $allArgs -join ' ' | Write-Host
    Write-Host ""

    try {
        Start-Process -FilePath $SERVER_PATH -ArgumentList $allArgs -NoNewWindow -Wait -RedirectStandardOutput $logFile
    } catch {
        Write-Host "ERROR: Failed to start server: $_" -ForegroundColor Red
        Read-Host "Press Enter to continue"
    }
}

do {
    Show-Menu
    $choice = Read-Host "Select a model [1-8, S, X]"

    switch ($choice) {
        'S' {
            if (Test-PortInUse $DEFAULT_PORT) {
                Stop-ServerOnPort $DEFAULT_PORT
                Write-Host "Server stopped." -ForegroundColor Green
            } else {
                Write-Host "No server running on port $DEFAULT_PORT" -ForegroundColor Yellow
            }
            Start-Sleep -Seconds 1
        }
        's' {
            if (Test-PortInUse $DEFAULT_PORT) {
                Stop-ServerOnPort $DEFAULT_PORT
                Write-Host "Server stopped." -ForegroundColor Green
            } else {
                Write-Host "No server running on port $DEFAULT_PORT" -ForegroundColor Yellow
            }
            Start-Sleep -Seconds 1
        }
        '1' {
            # GUIDE: https://huggingface.co/unsloth/Qwen3-Coder-30B-A3B-Instruct
            # Best Practices: temp=0.7, top_p=0.8, top_k=20, repetition_penalty=1.05
            Start-LLMServer -ModelPath $MODELS['1'].Path -Alias 'qwen3-coder' -Arguments @(
                '-m',        $MODELS['1'].Path
                '--host',    '0.0.0.0'
                '--port',    $DEFAULT_PORT
                '-fa',       'on'
                '-c',        '65536'
                '-ngl',      '99'
                '--mlock'
                '--n-cpu-moe','48'
                '-t',        '8'
                '-ctk',      'q4_0'
                '-ctv',      'q4_0'
                '-b',        '512'
                '-np',       '2'
                '--temp',    '0.7'
                '--top-p',   '0.8'
                '--top-k',   '20'
                '--repeat-penalty', '1.05'
                '-ot',       '.ffn_.*_exps.=CPU'
                '--jinja'
                '--alias',   'qwen3-coder'
            )
        }
        '2' {
            # GUIDE: https://huggingface.co/unsloth/Qwen3-30B-A3B-Thinking-2507-GGUF
            # Best Practices: temp=0.6, top_p=0.95, top_k=20, min_p=0
            Start-LLMServer -ModelPath $MODELS['2'].Path -Alias 'qwen3-thinking' -Arguments @(
                '-m',        $MODELS['2'].Path
                '--host',    '0.0.0.0'
                '--port',    $DEFAULT_PORT
                '-fa',       'on'
                '-c',        '32768'
                '-ngl',      '99'
                '--mlock'
                '--n-cpu-moe','48'
                '-t',        '8'
                '-ctk',      'q4_0'
                '-ctv',      'q4_0'
                '-b',        '512'
                '-np',       '2'
                '--temp',    '0.6'
                '--top-p',   '0.95'
                '--top-k',   '20'
                '--min-p',   '0'
                '-ot',       '.ffn_.*_exps.=CPU'
                '--jinja'
                '--alias',   'qwen3-thinking'
            )
        }
        '3' {
            # GUIDE: https://huggingface.co/unsloth/Qwen3-VL-30B-A3B-Instruct-GGUF
            # Vision model - keep temp low for accuracy
            Start-LLMServer -ModelPath $MODELS['3'].Path -Alias 'qwen3-vl' -Arguments @(
                '-m',        $MODELS['3'].Path
                '--mmproj',  "$LLM_DIR\Qwen3-VL-Instruct\mmproj-F16.gguf"
                '--host',    '0.0.0.0'
                '--port',    $DEFAULT_PORT
                '-fa',       'on'
                '-c',        '65536'
                '-ngl',      '99'
                '--mlock'
                '--n-cpu-moe','48'
                '-t',        '8'
                '-ctk',      'q4_0'
                '-ctv',      'q4_0'
                '--image-min-tokens', '1024'
                '--temp',    '0.1'
                '--top-p',   '0.95'
                '--top-k',   '20'
                '-ot',       '.ffn_.*_exps.=CPU'
                '--jinja'
                '--alias',   'qwen3-vl'
            )
        }
        '4' {
            # GUIDE: https://huggingface.co/unsloth/Nemotron-3-Nano-30B-A3B
            # NVIDIA recommends: temp=1.0, top_p=1.0 for reasoning; temp=0.6, top_p=0.95 for tool calling
            Start-LLMServer -ModelPath $MODELS['4'].Path -Alias 'nemotron' -Arguments @(
                '-m',        $MODELS['4'].Path
                '--host',    '0.0.0.0'
                '--port',    $DEFAULT_PORT
                '-fa',       'on'
                '-c',        '32768'
                '-ngl',      '99'
                '--mlock'
                '--n-cpu-moe','48'
                '-t',        '8'
                '-ctk',      'q4_0'
                '-ctv',      'q4_0'
                '-b',        '512'
                '-np',       '2'
                '--temp',    '1.0'
                '--top-p',   '1.0'
                '-ot',       '.ffn_.*_exps.=CPU'
                '--jinja'
                '--chat-template-kwargs', '{"enable_thinking":true}'
                '--alias',   'nemotron'
            )
        }
        '5' {
            # GUIDE: https://docs.unsloth.ai/models/glm-4.6-how-to-run-locally
            # Z.ai recommended: temp=0.8, top_p=0.6, top_k=2, repeat_penalty=1.1, min_p=0
            Start-LLMServer -ModelPath $MODELS['5'].Path -Alias 'glm-flash' -Arguments @(
                '-m',        $MODELS['5'].Path
                '--mmproj',  "$LLM_DIR\GLM-4.6V-Flash\mmproj-F16.gguf"
                '--host',    '0.0.0.0'
                '--port',    $DEFAULT_PORT
                '-fa',       'on'
                '-c',        '131072'
                '-ngl',      '99'
                '--mlock'
                '-t',        '8'
                '-ctk',      'q4_0'
                '-ctv',      'q4_0'
                '-b',        '512'
                '--temp',    '0.8'
                '--top-p',   '0.6'
                '--top-k',   '2'
                '--min-p',   '0.0'
                '--repeat-penalty', '1.1'
                '-ot',       '.ffn_.*_exps.=CPU'
                '--jinja'
                '--alias',   'glm-flash'
            )
        }
        '6' {
            # GUIDE: https://huggingface.co/unsloth/medgemma-1.5-4b-it
            Start-LLMServer -ModelPath $MODELS['6'].Path -Alias 'medgemma' -Arguments @(
                '-m',        $MODELS['6'].Path
                '--mmproj',  "$LLM_DIR\mmproj-F16.gguf"
                '--host',    '0.0.0.0'
                '--port',    $DEFAULT_PORT
                '-fa',       'on'
                '-c',        '16384'
                '-ngl',      '99'
                '--mlock'
                '-t',        '8'
                '-b',        '512'
                '-np',       '4'
                '--temp',    '0.2'
                '--jinja'
                '--alias',   'medgemma'
            )
        }
        '7' {
            # GUIDE: https://github.com/openai/gpt-oss
            # Heavy MoE model - offload experts to CPU for better GPU utilization
            Start-LLMServer -ModelPath $MODELS['7'].Path -Alias 'gpt-oss' -Arguments @(
                '-m',        $MODELS['7'].Path
                '--host',    '0.0.0.0'
                '--port',    $DEFAULT_PORT
                '-fa',       'on'
                '-c',        '32768'
                '-ngl',      '99'
                '--n-cpu-moe','62'
                '-t',        '8'
                '-ctk',      'q4_0'
                '-ctv',      'q4_0'
                '-b',        '512'
                '--temp',    '0.7'
                '--top-p',   '0.95'
                '--min-p',   '0.05'
                '-ot',       '.ffn_.*_exps.=CPU'
                '--jinja'
                '--alias',   'gpt-oss'
            )
        }
        '8' {
            # GUIDE: https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF
            Start-LLMServer -ModelPath $MODELS['8'].Path -Alias 'lfm-edge' -Arguments @(
                '-m',        $MODELS['8'].Path
                '--host',    '0.0.0.0'
                '--port',    $DEFAULT_PORT
                '-fa',       'on'
                '-c',        '32768'
                '-ngl',      '99'
                '--mlock'
                '-t',        '8'
                '-b',        '1024'
                '-ub',       '1024'
                '-np',       '4'
                '--temp',    '0.1'
                '--jinja'
                '--alias',   'lfm-edge'
            )
        }
        'X' { return }
        'x' { return }
    }
} while ($choice -ne 'X' -and $choice -ne 'x')

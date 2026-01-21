# LLM Manager
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
# | GPT-OSS (OpenAI)  | 1.0  | 1.0   | -     | 0.01  | -              |
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
# Nemotron: Use enable_thinking=true for reasoning, false for faster non-reasoning responses
# Qwen3-Thinking: Supports 256K context, recommend >131K for complex reasoning
# Vision models: Require --mmproj for the vision encoder
# GPT-OSS-120B: Heavy MoE model. Avoid KV cache quant (hurts perf).
#               min_p=0.0 causes 40% slowdown vs min_p=0.01
#
# DOCUMENTATION SOURCES:
# ----------------------
# Qwen3-Coder:    https://huggingface.co/unsloth/Qwen3-Coder-30B-A3B-Instruct
# Qwen3-Thinking: https://huggingface.co/unsloth/Qwen3-30B-A3B-Thinking-2507-GGUF
# Qwen3-VL:       https://huggingface.co/unsloth/Qwen3-VL-30B-A3B-Instruct-GGUF
# Nemotron:       https://huggingface.co/unsloth/Nemotron-3-Nano-30B-A3B
# GLM-4.7:        https://unsloth.ai/docs/models/glm-4.7-flash
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
$SERVER_PATH = "C:\Temp\llama.cpp\build\bin\llama-server.exe"
$LLM_DIR = "H:\LLM"
$DEFAULT_PORT = 8080
$LOG_DIR = "$PSScriptRoot\logs"
$FIT_MODE = 'on'  # set to 'off' to disable auto-fit
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
    $Host.UI.RawUI.WindowTitle = "LLM Manager"
} catch { }

# Model definitions with metadata
$MODELS = @{
    '1' = @{
        Name = "Qwen3-Coder-30B"; Context = "65k"; Category = "Coding Expert"
        Path = "C:\Temp\Qwen3-Coder-30B-A3B-Instruct-Q8_0.gguf"
        MoE = $true
    }
    '2' = @{
        Name = "Qwen3-30B Thinking"; Context = "32k"; Category = "Logic/Reasoning"
        Path = "$LLM_DIR\Qwen3-30B-A3B-UD-Q8_K_XL.gguf"
        MoE = $true
    }
    '3' = @{
        Name = "Qwen3-VL-30B"; Context = "65k"; Category = "Vision"
        Path = "$LLM_DIR\Qwen3-VL-Instruct\Qwen3-VL-30B-A3B-Instruct-UD-Q8_K_XL.gguf"
        MoE = $true
    }
    '4' = @{
        Name = "Nemotron-3-Nano"; Context = "32k"; Category = "Deep Thinking"
        Path = "$LLM_DIR\Nemotron-3-Nano-30B-A3B-UD-Q8_K_XL.gguf"
        MoE = $true
    }
    '5' = @{
        Name = "MedGemma-1.5-4B"; Context = "16k"; Category = "Medical"
        Path = "$LLM_DIR\MedGemma-1.5\MedGemma-1.5-4b-it-UD-Q8_K_XL.gguf"
        MoE = $false
    }
    '6' = @{
        Name = "GPT-OSS-120B"; Context = "32k"; Category = "Heavy MoE"
        Path = "C:\Temp\gpt-oss-120b-mxfp4-00001-of-00003.gguf"
        MoE = $true
    }
    '7' = @{
        Name = "LFM2.5-1.2B"; Context = "32k"; Category = "Liquid AI Edge"
        Path = "$LLM_DIR\LFM2.5-1.2B-Instruct-BF16.gguf"
        MoE = $false
    }
    '8' = @{
        Name = "GLM-4.7-Flash"; Context = "16k"; Category = "Fast General"
        Path = "$LLM_DIR\GLM-4.7-Flash-UD-Q4_K_XL.gguf"
        MoE = $false
    }
}

# Common CLI args to reduce duplication
$COMMON_ARGS = @(
    '--host',         '0.0.0.0'
    '--port',         $DEFAULT_PORT
    '--fit',          $FIT_MODE
    '--flash-attn',   'on'
    '--n-gpu-layers', '99'
    '--mlock'
    '--threads',      '8'
    '--jinja'
)

function Get-FileSize([string]$Path) {
    if (Test-Path $Path) {
        $size = (Get-Item $Path).Length / 1GB
        return "{0:N1}GB" -f $size
    }
    return "N/A"
}

function Get-ServerProcessesOnPort([int]$Port) {
    $conns = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue
    if (-not $conns) { return @() }
    $processIds = $conns | Select-Object -Unique -ExpandProperty OwningProcess
    $procs = @()
    foreach ($processId in $processIds) {
        $proc = Get-Process -Id $processId -ErrorAction SilentlyContinue
        if ($proc) { $procs += $proc }
    }
    return $procs
}

function Test-PortInUse([int]$Port) {
    return (Get-ServerProcessesOnPort -Port $Port).Count -gt 0
}

function Wait-ForPortOpen([int]$Port, [int]$TimeoutSeconds = 20) {
    $start = Get-Date
    while ((Get-Date) -lt $start.AddSeconds($TimeoutSeconds)) {
        if (Test-PortInUse $Port) { return $true }
        Start-Sleep -Milliseconds 250
    }
    return $false
}

function Wait-ForPortClose([int]$Port, [int]$TimeoutSeconds = 15) {
    $start = Get-Date
    while ((Get-Date) -lt $start.AddSeconds($TimeoutSeconds)) {
        if (-not (Test-PortInUse $Port)) { return $true }
        Start-Sleep -Milliseconds 250
    }
    return $false
}

function Write-PidFile([string]$Alias, [int]$ProcessId) {
    $pidFile = "$LOG_DIR\${Alias}.pid"
    $content = @{ pid = $ProcessId; started = (Get-Date).ToString('o') } | ConvertTo-Json
    Set-Content -Path $pidFile -Value $content -Encoding UTF8 -Force
}

function Read-PidFile([string]$Alias) {
    $pidFile = "$LOG_DIR\${Alias}.pid"
    if (Test-Path $pidFile) {
        try {
            $obj = Get-Content $pidFile -Raw | ConvertFrom-Json
            return $obj
        } catch { return $null }
    }
    return $null
}

function Remove-PidFile([string]$Alias) {
    $pidFile = "$LOG_DIR\${Alias}.pid"
    if (Test-Path $pidFile) { Remove-Item $pidFile -Force -ErrorAction SilentlyContinue }
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
        Write-Host -NoNewline "Size:$($size.PadRight(7))" -ForegroundColor DarkGray
        Write-Host $status -ForegroundColor $statusColor
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

    # 1. Validate model file exists
    if (-not (Test-Path $ModelPath)) {
        Write-Host ""
        Write-Host "ERROR: Model file not found!" -ForegroundColor Red
        Write-Host "Path: $ModelPath" -ForegroundColor Red
        Write-Host ""
        Read-Host "Press Enter to continue"
        return
    }

    # 2. Check for port conflicts (Zombie processes)
    if (Test-PortInUse $DEFAULT_PORT) {
        Write-Host "Port $DEFAULT_PORT is already in use. Checking for existing processes..." -ForegroundColor Yellow
        $procs = Get-ServerProcessesOnPort -Port $DEFAULT_PORT
        
        if ($procs) {
            foreach ($p in $procs) {
                Write-Host "Killing existing server process (PID: $($p.Id))..." -ForegroundColor DarkYellow
                Stop-Process -InputObject $p -Force -ErrorAction SilentlyContinue
            }
            # Wait for port to clear
            if (-not (Wait-ForPortClose -Port $DEFAULT_PORT -TimeoutSeconds 5)) {
                Write-Host "ERROR: Could not release port $DEFAULT_PORT. Please kill the process manually." -ForegroundColor Red
                Read-Host "Press Enter to continue"
                return
            }
            Write-Host "Port released." -ForegroundColor Green
        } else {
             Write-Host "WARNING: Port $DEFAULT_PORT is in use by a process we cannot identify or access." -ForegroundColor Red
             Write-Host "Please close the application using port $DEFAULT_PORT manually." -ForegroundColor Red
             Read-Host "Press Enter to continue"
             return
        }
    }

    # 3. Prepare Arguments
    # Append API key only if it exists
    if ($env:LLM_API_KEY) {
        $allArgs = @($Arguments + @('--api-key', $env:LLM_API_KEY))
    } else {
        $allArgs = $Arguments
    }

    # Generate log filename
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $logFile = "$LOG_DIR\${Alias}_${timestamp}.log"
    
    Write-Host ""
    Write-Host "==========================================================" -ForegroundColor Green
    Write-Host "Starting: $Alias" -ForegroundColor Green
    Write-Host "Model: $ModelPath" -ForegroundColor DarkGray
    Write-Host "Log: $logFile" -ForegroundColor DarkGray
    Write-Host "==========================================================" -ForegroundColor Green
    Write-Host ""

    Write-Host "Server arguments:" -ForegroundColor Cyan
    $allArgs -join ' ' | Write-Host
    $cmdLine = '"{0}" {1}' -f $SERVER_PATH, (($allArgs | ForEach-Object {
        if ($_ -match '\s') { '"{0}"' -f $_ } else { $_ }
    }) -join ' ')
    Write-Host "Command line:" -ForegroundColor Cyan
    Write-Host $cmdLine
    Write-Host ""
    
    # 4. Start Server
    # We use Start-Process to keep it clean, but wait-process to block 
    # so the user can see the output in this window if they want, 
    # OR we can run it inside the current console. 
    # Given the 'Manager' nature, running inside the console with '&' is best for visibility,
    # but we wrap it to ensure we catch exit codes.
    
    try {
        # Track PID for "Manager" logic if we decide to go non-blocking later
        Write-PidFile -Alias $Alias -ProcessId $PID 
        
        # Run the server (Blocking)
        & $SERVER_PATH @($allArgs) 2>&1 | Tee-Object -FilePath $logFile
        
    } catch {
        Write-Host "ERROR: Server crashed or failed to start: $_" -ForegroundColor Red
    } finally {
        Remove-PidFile -Alias $Alias
        Write-Host ""
        Write-Host "Server stopped." -ForegroundColor Yellow
        Read-Host "Press Enter to return to menu"
    }
}

do {
    # -------------------------------------------------------------------------
    # GLOBAL OPTIMIZATION NOTE (Ryzen 5900HX + RTX 3070 8GB + 64GB RAM):
    # MoE Models (Qwen, GLM, GPT-OSS) use `-ot .ffn_.*_exps.=CPU` or `--n-cpu-moe`
    # to offload the massive "Sparse Expert" layers to system RAM while keeping
    # the active "hot" layers on the GPU. This allows running 100B+ models on 8GB VRAM.
    # -------------------------------------------------------------------------

    Show-Menu
    $choice = Read-Host "Select a model [1-8, X]"

    switch ($choice) {
        '1' {
            # GUIDE: https://huggingface.co/unsloth/Qwen3-Coder-30B-A3B-Instruct
            # Best Practices: temp=0.7, top_p=0.8, top_k=20, repetition_penalty=1.05
            Start-LLMServer -ModelPath $MODELS['1'].Path -Alias 'qwen3-coder' -Arguments @(
                $COMMON_ARGS + @(
                '--model',        $MODELS['1'].Path
                '--ctx-size',     '65536'
                '--n-cpu-moe',    '48'
                '--cache-type-k', 'q4_0'
                '--cache-type-v', 'q4_0'
                '--batch-size',   '512'
                '--parallel',     '2'
                '--temp',         '0.7'
                '--top-p',        '0.8'
                '--top-k',        '20'
                '--repeat-penalty', '1.05'
                '-ot',            '.ffn_.*_exps.=CPU'
                '--alias',        'qwen3-coder'
                )
            )
        }
        '2' {
            # GUIDE: https://huggingface.co/unsloth/Qwen3-30B-A3B-Thinking-2507-GGUF
            # Best Practices: temp=0.6, top_p=0.95, top_k=20, min_p=0
            # CRITICAL: min_p=0 prevents cutting off low-probability reasoning tokens.
            Start-LLMServer -ModelPath $MODELS['2'].Path -Alias 'qwen3-thinking' -Arguments @(
                $COMMON_ARGS + @(
                '--model',        $MODELS['2'].Path
                '--ctx-size',     '65536'
                '--n-cpu-moe',    '48'
                '--cache-type-k', 'q4_0'
                '--cache-type-v', 'q4_0'
                '--batch-size',   '512'
                '--parallel',     '2'
                '--temp',         '0.6'
                '--top-p',        '0.95'
                '--top-k',        '20'
                '--min-p',        '0'
                '-ot',            '.ffn_.*_exps.=CPU'
                '--chat-template-kwargs', '{"enable_thinking":true}'
                '--alias',        'qwen3-thinking'
                )
            )
        }
        '3' {
            # GUIDE: https://huggingface.co/unsloth/Qwen3-VL-30B-A3B-Instruct-GGUF
            # Vision model - keep temp low for accuracy
            Start-LLMServer -ModelPath $MODELS['3'].Path -Alias 'qwen3-vl' -Arguments @(
                $COMMON_ARGS + @(
                '--model',        $MODELS['3'].Path
                '--mmproj',       "$LLM_DIR\Qwen3-VL-Instruct\mmproj-F16.gguf"
                '--ctx-size',     '131072'
                '--n-cpu-moe',    '48'
                '--cache-type-k', 'q4_0'
                '--cache-type-v', 'q4_0'
                '--batch-size',   '512'
                '--parallel',     '2'
                '--image-min-tokens', '1024'
                '--temp',         '0.1'
                '--top-p',        '0.95'
                '--top-k',        '20'
                '-ot',            '.ffn_.*_exps.=CPU'
                '--alias',        'qwen3-vl'
                )
            )
        }
        '4' {
            # GUIDE: https://huggingface.co/unsloth/Nemotron-3-Nano-30B-A3B
            # NVIDIA recommends: temp=1.0, top_p=1.0 for reasoning; temp=0.6, top_p=0.95 for tool calling
            $kwargs = '{"enable_thinking":true}'
            Start-LLMServer -ModelPath $MODELS['4'].Path -Alias 'nemotron' -Arguments @(
                $COMMON_ARGS + @(
                '--model',        $MODELS['4'].Path
                '--ctx-size',     '32768'
                '--n-cpu-moe',    '48'
                '--cache-type-k', 'q4_0'
                '--cache-type-v', 'q4_0'
                '--batch-size',   '512'
                '--parallel',     '2'
                '--temp',         '1.0'
                '--top-p',        '1.0'
                '-ot',            '.ffn_.*_exps.=CPU'
                '--chat-template-kwargs', $kwargs
                '--alias',        'nemotron'
                )
            )
        }
        '5' {
            # GUIDE: https://huggingface.co/unsloth/medgemma-1.5-4b-it
            Start-LLMServer -ModelPath $MODELS['5'].Path -Alias 'medgemma' -Arguments @(
                $COMMON_ARGS + @(
                '--model',        $MODELS['5'].Path
                '--mmproj',       "$LLM_DIR\MedGemma-1.5\mmproj-F16.gguf"
                '--ctx-size',     '16384'
                '--batch-size',   '512'
                '--parallel',     '4'
                '--temp',         '0.2'
                '--alias',        'medgemma'
                )
            )
        }
        '6' {
            # GUIDE: https://github.com/ggml-org/llama.cpp/discussions/15396
            # GPT-OSS-120B: Heavy MoE model (requires aggressive offloading)
            # CRITICAL: Top-K forced to 0 (disabled) to avoid performance kill.
            # OPTIMIZATION: min_p=0 for accuracy (slower) vs min_p=0.01 for speed.
            Start-LLMServer -ModelPath $MODELS['6'].Path -Alias 'gpt-oss' -Arguments @(
                $COMMON_ARGS + @(
                '--model',        $MODELS['6'].Path
                '--ctx-size',     '32768'
                '--no-mmap'
                '--n-cpu-moe',    '35'
                '--batch-size',   '2048'
                '--ubatch-size',  '2048'
                '--temp',         '1.0'
                '--top-p',        '1.0'
                '--top-k',        '0'
                '--min-p',        '0'
                '--alias',        'gpt-oss'
                )
            )
        }
        '7' {
            # GUIDE: https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF
            # Edge Model (1.2B). Optimized for low-latency on-device use.
            # Best Practice: Keep temperature low (0.1) for stability.
            Start-LLMServer -ModelPath $MODELS['7'].Path -Alias 'lfm-edge' -Arguments @(
                $COMMON_ARGS + @(
                '--model',        $MODELS['7'].Path
                '--ctx-size',     '32768'
                '--batch-size',   '1024'
                '--ubatch-size',  '1024'
                '--parallel',     '4'
                '--temp',         '0.1'
                '--alias',        'lfm-edge'
                )
            )
        }
        '8' {
            Start-LLMServer -ModelPath $MODELS['8'].Path -Alias 'glm4.7-flash' -Arguments @(
                $COMMON_ARGS + @(
                '--model',        $MODELS['8'].Path
                '--ctx-size',     '16384'
                '--batch-size',   '1024'
                '--ubatch-size',  '512'
                '--temp',         '0.7'
                '--top-p',        '1.0'
                '--min-p',        '0.01'
                '-ot',            '.ffn_.*_exps.=CPU'
                '--alias',        'glm4.7-flash'
                )
            )
        }
        { $_ -eq 'X' -or $_ -eq 'x' } { return }
    }
} while ($choice -notin @('X', 'x'))

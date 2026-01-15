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
# | GLM-4.6V-Flash    | 0.8  | 0.6   | 2     | 0.0   | 1.1            |
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
# GLM-4.6V: May reason in Chinese. Use: --system-prompt "Respond in English and reason in English"
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
$SERVER_PATH = "C:\Temp\llama.cpp\build\bin\llama-server.exe"
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
        Name = "GLM-4.6V-Flash"; Context = "131k"; Category = "Ultra Vision/Fast"
        Path = "$LLM_DIR\GLM-4.6V-Flash\GLM-4.6V-Flash-UD-Q4_K_XL.gguf"
        MoE = $true
    }
    '6' = @{
        Name = "MedGemma-1.5-4B"; Context = "16k"; Category = "Medical"
        Path = "$LLM_DIR\MedGemma-1.5\MedGemma-1.5-4b-it-UD-Q8_K_XL.gguf"
        MoE = $false
    }
    '7' = @{
        Name = "GPT-OSS-120B"; Context = "32k"; Category = "Heavy MoE"
        Path = "C:\Temp\gpt-oss-120b-mxfp4-00001-of-00003.gguf"
        MoE = $true
    }
    '8' = @{
        Name = "LFM2.5-1.2B"; Context = "32k"; Category = "Liquid AI Edge"
        Path = "$LLM_DIR\LFM2.5-1.2B-Instruct-BF16.gguf"
        MoE = $false
    }
}

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

function Stop-ProcessGraceful([System.Diagnostics.Process]$proc, [int]$TimeoutSeconds = 5) {
    if (-not $proc) { return $false }
    if ($proc.HasExited) { return $true }

    Write-Host "Attempting graceful stop of $($proc.ProcessName) (PID: $($proc.Id))..." -ForegroundColor Yellow
    try {
        # Try asking the process to exit cleanly
        if ($proc.MainWindowHandle -ne 0) {
            $proc.CloseMainWindow() | Out-Null
            $waitStart = Get-Date
            while (-not $proc.HasExited -and ((Get-Date) -lt $waitStart.AddSeconds($TimeoutSeconds))) {
                Start-Sleep -Milliseconds 200
                $proc.Refresh()
            }
            if ($proc.HasExited) { return $true }
        }

        # Fallback: try Stop-Process (gentle)
        Stop-Process -Id $proc.Id -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 1
        $proc.Refresh()
        if ($proc.HasExited) { return $true }

        # Force kill as last resort
        Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 1
        $proc.Refresh()
        return -not $proc.HasExited
    } catch {
        return $false
    }
}

function Stop-ServerOnPort([int]$Port) {
    $procs = Get-ServerProcessesOnPort -Port $Port
    if (-not $procs -or $procs.Count -eq 0) { return $false }

    $stoppedAny = $false
    foreach ($proc in $procs) {
        # Do not blindly kill unrelated processes - ensure it's our server executable before forcing
        try {
            if ($proc.Path -and ($proc.Path -ieq $SERVER_PATH -or ($proc.ProcessName -like '*llama*' -or $proc.ProcessName -like '*server*'))) {
                $ok = Stop-ProcessGraceful -proc $proc -TimeoutSeconds 6
                if ($ok) { $stoppedAny = $true }
            } else {
                # Attempt gentle stop but avoid force-killing system processes
                $ok = Stop-ProcessGraceful -proc $proc -TimeoutSeconds 4
                if ($ok) { $stoppedAny = $true }
            }
        } catch {
            # ignore and continue
        }
    }

    # Wait for port to close
    if (Wait-ForPortClose -Port $Port -TimeoutSeconds 10) { return $stoppedAny }

    # If still open, try force-killing any remaining owners
    $procs = Get-ServerProcessesOnPort -Port $Port
    foreach ($proc in $procs) {
        try {
            Write-Host "Force killing $($proc.ProcessName) (PID: $($proc.Id)) on port $Port..." -ForegroundColor Red
            Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
            $stoppedAny = $true
        } catch { }
    }

    return $stoppedAny
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
    
    Write-Host "----------------------------------------------------------"
    if (Test-PortInUse $DEFAULT_PORT) {
        $procs = Get-ServerProcessesOnPort $DEFAULT_PORT
        if ($procs -and $procs.Count -gt 0) {
            $p = $procs[0]
            Write-Host "S) Stop server on port $DEFAULT_PORT - $($p.ProcessName) (PID: $($p.Id))" -ForegroundColor Red
        } else {
            Write-Host "S) Stop server on port $DEFAULT_PORT" -ForegroundColor Red
        }
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

    # Check for existing pid file and running process
    $existing = Read-PidFile -Alias $Alias
    if ($existing -and $existing.pid) {
        try {
            $p = Get-Process -Id $existing.pid -ErrorAction SilentlyContinue
            if ($p -and -not $p.HasExited) {
                Write-Host "Detected existing PID file for $Alias -> PID $($existing.pid)." -ForegroundColor Yellow
                $resp = Read-Host "Stop existing process? [Y/N]"
                if ($resp -eq 'Y' -or $resp -eq 'y') {
                    Stop-ProcessGraceful -proc $p -TimeoutSeconds 6
                    Remove-PidFile -Alias $Alias
                } else {
                    Write-Host "Aborting start." -ForegroundColor Yellow
                    return
                }
            } else {
                Remove-PidFile -Alias $Alias
            }
        } catch { Remove-PidFile -Alias $Alias }
    }

    # Check if port is in use
    if (Test-PortInUse $DEFAULT_PORT) {
        Write-Host ""
        Write-Host "WARNING: Port $DEFAULT_PORT is already in use!" -ForegroundColor Yellow
        $response = Read-Host "Stop existing server? [Y/N]"
        if ($response -eq 'Y' -or $response -eq 'y') {
            if (-not (Stop-ServerOnPort $DEFAULT_PORT)) {
                Write-Host "Failed to stop existing server on port $DEFAULT_PORT. Aborting." -ForegroundColor Red
                return
            }
        } else {
            Write-Host "Aborting start due to port conflict." -ForegroundColor Yellow
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
        $proc = Start-Process -FilePath $SERVER_PATH -ArgumentList $allArgs -NoNewWindow -RedirectStandardOutput $logFile -RedirectStandardError $logFile -PassThru
        if (-not $proc) { throw "Failed to obtain process object." }

        # Write pidfile
        Write-PidFile -Alias $Alias -ProcessId $proc.Id

        # Wait for server to bind to port
        if (Wait-ForPortOpen -Port $DEFAULT_PORT -TimeoutSeconds 20) {
            Write-Host "Server started successfully: PID $($proc.Id)" -ForegroundColor Green
            return
        } else {
            Write-Host "ERROR: Server did not bind to port $DEFAULT_PORT within timeout." -ForegroundColor Red
            # Capture last lines of log for debugging
            try {
                Write-Host "----- Last 50 lines of log ($logFile) -----" -ForegroundColor DarkGray
                Get-Content -Path $logFile -Tail 50 | ForEach-Object { Write-Host $_ }
            } catch { }

            # Attempt to stop process
            if (-not $proc.HasExited) {
                Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
            }
            Remove-PidFile -Alias $Alias
            Read-Host "Press Enter to continue"
            return
        }
    } catch {
        Write-Host "ERROR: Failed to start server: $_" -ForegroundColor Red
        Read-Host "Press Enter to continue"
    }
}

do {
    Show-Menu
    $choice = Read-Host "Select a model [1-8, S, X]"

    switch ($choice) {
        { $_ -eq 'S' -or $_ -eq 's' } {
            if (Test-PortInUse $DEFAULT_PORT) {
                if (Stop-ServerOnPort $DEFAULT_PORT) {
                    Write-Host "Server stopped." -ForegroundColor Green
                } else {
                    Write-Host "WARNING: Stop attempt failed or required force. Please verify." -ForegroundColor Red
                }

                # Clean up stale PID files pointing to non-running processes
                Get-ChildItem -Path $LOG_DIR -Filter '*.pid' -ErrorAction SilentlyContinue | ForEach-Object {
                    try {
                        $json = Get-Content $_.FullName -Raw | ConvertFrom-Json
                        if (-not (Get-Process -Id $json.pid -ErrorAction SilentlyContinue)) {
                            Remove-Item $_.FullName -Force -ErrorAction SilentlyContinue
                        }
                    } catch { }
                }
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
                '-b',        '512'
                '-np',       '2'
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
            $kwargs = '{"enable_thinking":true}'
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
                '--chat-template-kwargs', $kwargs
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
                '--n-cpu-moe','48'
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
                '--mmproj',  "$LLM_DIR\MedGemma-1.5\mmproj-F16.gguf"
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
            # GUIDE: https://github.com/ggml-org/llama.cpp/discussions/15396
            # GPT-OSS-120B: Heavy MoE model
            # OpenAI recommends: temp=1.0, top_p=1.0; use min_p=0.01 for performance
            Start-LLMServer -ModelPath $MODELS['7'].Path -Alias 'gpt-oss' -Arguments @(
                '-m',        $MODELS['7'].Path
                '--host',    '0.0.0.0'
                '--port',    $DEFAULT_PORT
                '-fa',       'on'
                '-c',        '32768'
                '-ngl',      '99'
                '--mlock'
                '--no-mmap'
                '--n-cpu-moe','35'
                '-t',        '8'
                '-b',        '2048'
                '-ub',       '2048'
                '--temp',    '1.0'
                '--top-p',   '1.0'
                '--min-p',   '0.01'
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
        { $_ -eq 'X' -or $_ -eq 'x' } { return }
    }
} while ($choice -notin @('X', 'x'))

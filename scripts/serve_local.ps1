# Start llama.cpp's server for one F0 model, bound to this machine only.
# Binaries and weights live outside OneDrive in %LOCALAPPDATA%\gene2disease.
# Requests need the key in %LOCALAPPDATA%\gene2disease\api_key.txt, created on first
# start, so a web page open in a local browser cannot use the server.
param(
    [Parameter(Mandatory = $true)][ValidateSet('qwen3.5-9b', 'gemma-4-12b')][string]$Model,
    [int]$Port = 8081,
    # 32k holds every development window in F0 (up to 144 pairs, about 21k tokens with
    # the 400 + 100 x pairs output budget). The largest task-paper window (240 pairs) needs
    # 24.4k output tokens plus its prompt, so it may not fit; that is unresolved (D17)
    # until its prompt length and the memory at a larger context are measured.
    [int]$Context = 32768,
    # 'off' for the main runs; 'on' only for the F0 thinking comparison.
    [ValidateSet('off', 'on')][string]$Reasoning = 'off'
)
$ErrorActionPreference = 'Stop'
$root = Join-Path $env:LOCALAPPDATA 'gene2disease'
$server = Join-Path $root 'llama.cpp\b12182\llama-server.exe'
$weights = @{
    'qwen3.5-9b'  = 'Qwen3.5-9B-Q4_K_M.gguf'
    'gemma-4-12b' = 'gemma-4-12b-it-qat-q4_0.gguf'
}
$file = Join-Path $root ('downloads\' + $weights[$Model])
foreach ($path in $server, $file) {
    if (-not (Test-Path $path)) { throw "missing: $path" }
}
$keyFile = Join-Path $root 'api_key.txt'
if (-not (Test-Path $keyFile)) {
    # These calls also work in Windows PowerShell 5.1.
    $bytes = New-Object byte[] 24
    [System.Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
    $hex = [System.BitConverter]::ToString($bytes).Replace('-', '').ToLower()
    [System.IO.File]::WriteAllText($keyFile, $hex)
}
# One slot keeps per-call timing comparable.
& $server --host 127.0.0.1 --port $Port --model $file --alias $Model `
    --ctx-size $Context --n-gpu-layers 99 --parallel 1 --flash-attn on `
    --reasoning $Reasoning --api-key-file $keyFile --no-webui

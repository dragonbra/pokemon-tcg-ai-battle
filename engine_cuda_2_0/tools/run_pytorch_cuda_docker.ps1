[CmdletBinding()]
param(
  [Parameter(Position = 0, ValueFromRemainingArguments = $true)]
  [string[]]$Command,
  [string]$Image = "pytorch/pytorch:2.10.0-cuda13.0-cudnn9-runtime",
  [string]$RepoRoot = "",
  [string]$PipCache = ".pip-cache-docker",
  [switch]$Devel
)

$ErrorActionPreference = "Stop"

if ($Devel) {
  $Image = "pytorch/pytorch:2.10.0-cuda13.0-cudnn9-devel"
}

if (-not $RepoRoot) {
  $RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
} else {
  $RepoRoot = (Resolve-Path $RepoRoot).Path
}

$cachePath = if ([IO.Path]::IsPathRooted($PipCache)) {
  [IO.Path]::GetFullPath($PipCache)
} else {
  [IO.Path]::GetFullPath((Join-Path $RepoRoot $PipCache))
}
New-Item -ItemType Directory -Force -Path $cachePath | Out-Null

if (-not $Command -or $Command.Count -eq 0) {
  $Command = @(
    "python",
    "-c",
    "import importlib.util, torch, numpy; print('torch', torch.__version__, 'cuda', torch.version.cuda); print('cuda_available', torch.cuda.is_available()); print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None); print('numpy', numpy.__version__); print('mkl_fft_spec', importlib.util.find_spec('mkl_fft'))"
  )
}

$dockerArgs = @(
  "run",
  "--rm",
  "--gpus",
  "all",
  "-e",
  "PYTHONUNBUFFERED=1",
  "-e",
  "PIP_CACHE_DIR=/root/.cache/pip",
  "-v",
  "${RepoRoot}:/workspace",
  "-v",
  "${cachePath}:/root/.cache/pip",
  "-w",
  "/workspace",
  $Image
) + $Command

docker @dockerArgs

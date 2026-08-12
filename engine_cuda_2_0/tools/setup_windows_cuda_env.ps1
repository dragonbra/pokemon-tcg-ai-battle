[CmdletBinding()]
param(
  [string]$Environment = ".venv-cuda",
  [string]$BasePython = "python",
  [string]$NumpyVersion = "2.3.5",
  [switch]$RequireCuda
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$environmentPath = if ([IO.Path]::IsPathRooted($Environment)) {
  [IO.Path]::GetFullPath($Environment)
} else {
  [IO.Path]::GetFullPath((Join-Path $repoRoot $Environment))
}
$environmentPython = Join-Path $environmentPath "Scripts\python.exe"
$verifier = Join-Path $PSScriptRoot "verify_windows_openmp_runtime.py"

if (-not (Test-Path -LiteralPath $environmentPython)) {
  Write-Host "Creating Windows CUDA environment: $environmentPath"
  & $BasePython -m venv --system-site-packages $environmentPath
  if ($LASTEXITCODE -ne 0) {
    throw "Failed to create the virtual environment."
  }
}

# The base Conda NumPy is linked against MKL and loads Conda's libiomp5md.dll.
# The pip PyTorch wheel bundles a second libiomp5md.dll. A local PyPI NumPy
# wheel uses OpenBLAS and leaves PyTorch as the only OpenMP runtime owner.
& $environmentPython -m pip install `
  --disable-pip-version-check `
  --only-binary=:all: `
  --no-deps `
  --ignore-installed `
  "numpy==$NumpyVersion"
if ($LASTEXITCODE -ne 0) {
  throw "Failed to install the project-local OpenBLAS NumPy wheel."
}

Remove-Item Env:KMP_DUPLICATE_LIB_OK -ErrorAction SilentlyContinue
Remove-Item Env:MKL_THREADING_LAYER -ErrorAction SilentlyContinue
$cudaArgs = @()
if ($RequireCuda) {
  $cudaArgs += "--require-cuda"
}

& $environmentPython $verifier --order numpy-first @cudaArgs
if ($LASTEXITCODE -ne 0) {
  throw "NumPy-first OpenMP verification failed."
}
& $environmentPython $verifier --order torch-first @cudaArgs
if ($LASTEXITCODE -ne 0) {
  throw "Torch-first OpenMP verification failed."
}

Write-Host ""
Write-Host "Windows CUDA environment is ready."
Write-Host "Activate it with:"
Write-Host "  $environmentPath\Scripts\Activate.ps1"

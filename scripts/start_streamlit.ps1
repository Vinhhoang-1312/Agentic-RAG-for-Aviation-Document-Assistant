param(
    [int]$Port = 8501
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Python virtual environment not found at $python"
}

& $python -m streamlit run streamlit_app.py --server.headless true --server.address 127.0.0.1 --server.port $Port

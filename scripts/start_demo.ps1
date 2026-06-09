param(
    [int]$ApiPort = 8000,
    [int]$UiPort = 8501
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Python virtual environment not found at $python"
}

$apiCommand = @"
Set-Location '$root'
& '$python' -m uvicorn 'aviation_rag.api:app' --host 127.0.0.1 --port $ApiPort --log-level info
"@

$uiCommand = @"
Set-Location '$root'
& '$python' -m streamlit run streamlit_app.py --server.headless true --server.address 127.0.0.1 --server.port $UiPort
"@

Start-Process powershell.exe -ArgumentList @(
    "-NoExit",
    "-ExecutionPolicy", "Bypass",
    "-Command", $apiCommand
) -WorkingDirectory $root

Start-Process powershell.exe -ArgumentList @(
    "-NoExit",
    "-ExecutionPolicy", "Bypass",
    "-Command", $uiCommand
) -WorkingDirectory $root

Write-Host "API: http://127.0.0.1:$ApiPort/health"
Write-Host "UI:  http://127.0.0.1:$UiPort"

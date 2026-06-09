param(
    [int]$ApiPort = 8000,
    [int]$UiPort = 8501
)

$ErrorActionPreference = "Stop"

try {
    $health = Invoke-WebRequest -Uri "http://127.0.0.1:$ApiPort/health" -UseBasicParsing -TimeoutSec 10
    Write-Host "API OK:" $health.Content
} catch {
    Write-Host "API FAIL:" $_.Exception.Message
}

try {
    $ui = Invoke-WebRequest -Uri "http://127.0.0.1:$UiPort" -UseBasicParsing -TimeoutSec 10
    Write-Host "UI OK:" $ui.StatusCode
} catch {
    Write-Host "UI FAIL:" $_.Exception.Message
}

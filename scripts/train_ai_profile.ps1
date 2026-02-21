param(
    [string]$OutputPath = "data/ai_training/profile.json"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

python scripts\ai_audio_bridge_server.py --learn-export $OutputPath

if ($LASTEXITCODE -ne 0) {
  exit $LASTEXITCODE
}

Write-Host "Profile exported to $OutputPath"

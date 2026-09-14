# ============================================================
# report.ps1 — report launcher for object_detection_tracking
# Run from:  PS C:\Users\shiva\OneDrive\Desktop\task 4>
# Usage:
#   .\report.ps1                       # most recent session
#   .\report.ps1 --all-sessions        # all sessions
#   .\report.ps1 --session 3           # specific session ID
# ============================================================

$projectDir = Join-Path $PSScriptRoot "object_detection_tracking"
Set-Location $projectDir
python report.py @args

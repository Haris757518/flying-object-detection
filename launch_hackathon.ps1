param([string]$Python = 'python')
$ErrorActionPreference = 'Stop'
Push-Location (Join-Path $PSScriptRoot 'code\hackathon_app')
try { & $Python -m streamlit run dashboard.py } finally { Pop-Location }

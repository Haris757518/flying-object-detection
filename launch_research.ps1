param([string]$Module = '9', [string]$Python = 'python')
$ErrorActionPreference = 'Stop'
Push-Location (Join-Path $PSScriptRoot 'code\research_prototype')
try { & $Python main.py $Module } finally { Pop-Location }

$ErrorActionPreference = 'Continue'
$base = Split-Path -Parent $MyInvocation.MyCommand.Path
$dest = Join-Path $base 'downloaded_pdfs'
New-Item -ItemType Directory -Force -Path $dest | Out-Null
$items = Import-Csv -Path (Join-Path $base 'papers_30_manifest.csv')
$results = @()
foreach ($r in $items) {
  $safe = ($r.title -replace '[^A-Za-z0-9]+','_')
  if ($safe.Length -gt 57) { $safe = $safe.Substring(0,57) }
  $num = '{0:D2}' -f [int]$r.number
  $target = Join-Path $dest ($num + '_' + $safe.Trim('_') + '.pdf')
  if (Test-Path $target) {
    $results += [PSCustomObject]@{number=$r.number;status='EXISTS';path=$target;source=$r.pdf_url}
    continue
  }
  try {
    Write-Host ('Downloading [' + $num + '/30]: ' + $r.title)
    Invoke-WebRequest -Uri $r.pdf_url -OutFile $target -UseBasicParsing -TimeoutSec 120 -MaximumRedirection 10 -Headers @{'User-Agent'='Mozilla/5.0 (Research literature collection)'}
    $stream=[IO.File]::OpenRead($target);$buffer=New-Object byte[] 5;$count=$stream.Read($buffer,0,5);$stream.Close()
    $magic=[Text.Encoding]::ASCII.GetString($buffer)
    if ($count -lt 5 -or $magic -ne '%PDF-') { throw ('Not a valid PDF: ' + $magic) }
    $results += [PSCustomObject]@{number=$r.number;status='DOWNLOADED';path=$target;source=$r.pdf_url}
  } catch {
    Write-Warning ('Failed ['+$num+']: '+$_.Exception.Message)
    if (Test-Path $target) { Remove-Item $target -Force }
    $results += [PSCustomObject]@{number=$r.number;status='FAILED';path='';source=$r.pdf_url}
  }
}
$results | Export-Csv -NoTypeInformation -Encoding UTF8 -Path (Join-Path $base 'download_status.csv')
$count=(Get-ChildItem -Path $dest -Filter '*.pdf' | Measure-Object).Count
Write-Host ('Original PDFs present: ' + $count + '/30')
if ($count -eq 30) {
  $zip=Join-Path $base 'All_30_Research_Papers_Complete.zip'
  if (Test-Path $zip) { Remove-Item $zip -Force }
  Compress-Archive -Path (Join-Path $dest '*.pdf') -DestinationPath $zip
  Write-Host ('Created complete ZIP: '+$zip)
} else { Write-Host 'Open the official publisher PDF links for any failed downloads; no placeholder PDFs were created.' }

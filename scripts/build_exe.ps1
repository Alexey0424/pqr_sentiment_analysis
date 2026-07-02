# Build the standalone Windows distribution of PQR Sentiment Studio.
# Requires the full stack: pip install -e ".[ml,app]" pyinstaller
# Output: release/PQR_Sentiment_Studio.zip (folder build, first run downloads the models)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

pyinstaller packaging/pqr_studio.spec --distpath release/dist --workpath release/build --noconfirm

$source = "release/dist/PQR_Sentiment_Studio"
$zip = "release/PQR_Sentiment_Studio.zip"
if (Test-Path $zip) { Remove-Item $zip -Force }
Compress-Archive -Path $source -DestinationPath $zip

Write-Output "Built $zip"

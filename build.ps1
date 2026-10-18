# Build Local Coder into a standalone Windows app with PyInstaller.
#
#   .\build.ps1                 -> dist\LocalCoder\LocalCoder.exe
#   .\build.ps1 -Name LocalCoder-dev
#
# The result is a folder (onedir) because imgui-bundle ships native binaries
# and assets that are far more reliable unpacked than squeezed into one file.
# The model weights and the llama.cpp build are NOT bundled: put `models\`
# and `llama\` next to the .exe (inside the dist\<Name>\ folder), exactly as
# they sit next to run.py when running from source.

param([string]$Name = "LocalCoder")
Set-Location $PSScriptRoot

# Everything goes through `python -m` so nothing depends on the Scripts dir
# being on PATH. pip prints its "not on PATH" note to stderr, which PowerShell
# 5.1 would otherwise turn into a terminating error, so we don't stop on it.
Write-Host "Installing build dependencies..." -ForegroundColor Cyan
python -m pip install --quiet --upgrade --no-warn-script-location pyinstaller -r requirements.txt
if ($LASTEXITCODE -ne 0) { Write-Host "pip install failed" -ForegroundColor Red; exit 1 }

Write-Host "Building $Name ..." -ForegroundColor Cyan
python -m PyInstaller --noconfirm --clean --windowed --name $Name `
    --collect-all imgui_bundle `
    --collect-all vertexui `
    --hidden-import OpenGL --hidden-import OpenGL.GL `
    --hidden-import OpenGL.platform.win32 `
    --hidden-import PIL --hidden-import PIL.Image --hidden-import PIL.ImageGrab `
    --hidden-import numpy `
    run.py
if ($LASTEXITCODE -ne 0) { Write-Host "PyInstaller failed" -ForegroundColor Red; exit 1 }

Write-Host ""
Write-Host "Done -> dist\$Name\$Name.exe" -ForegroundColor Green
Write-Host "Copy your 'llama\' and 'models\' folders into dist\$Name\ before running." -ForegroundColor Yellow

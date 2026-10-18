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
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "Installing build dependencies..." -ForegroundColor Cyan
python -m pip install --quiet --upgrade pyinstaller -r requirements.txt

Write-Host "Building $Name ..." -ForegroundColor Cyan
pyinstaller --noconfirm --clean --windowed --name $Name `
    --collect-all imgui_bundle `
    --collect-all vertexui `
    --hidden-import OpenGL --hidden-import OpenGL.GL `
    --hidden-import OpenGL.platform.win32 `
    --hidden-import PIL --hidden-import PIL.Image --hidden-import PIL.ImageGrab `
    --hidden-import numpy `
    run.py

if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed" }
Write-Host ""
Write-Host "Done -> dist\$Name\$Name.exe" -ForegroundColor Green
Write-Host "Copy your 'llama\' and 'models\' folders into dist\$Name\ before running." -ForegroundColor Yellow

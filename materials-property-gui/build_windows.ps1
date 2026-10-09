$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

python -m pip install -r requirements-build.txt
if ($LASTEXITCODE -ne 0) {
    throw "Failed to install the Windows build requirements."
}

python -m PyInstaller `
    --noconfirm `
    --clean `
    --windowed `
    --onedir `
    --name MaterialsPropertyGUI `
    --paths src `
    --add-data "src\data;src\data" `
    --add-data "src\dft\run.sh;src\dft" `
    --collect-all torch `
    --collect-all mace `
    --collect-all e3nn `
    --collect-all ase `
    --collect-all paramiko `
    src\main.py

if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller failed to build MaterialsPropertyGUI."
}

Write-Host "Build complete: dist\MaterialsPropertyGUI\MaterialsPropertyGUI.exe"
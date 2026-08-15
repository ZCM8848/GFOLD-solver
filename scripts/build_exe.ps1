# 打包数据生成器为 exe（onedir：多进程 spawn 需要，比 onefile 启动快得多）
# 产物在 dist/tofnet_generate/，整个文件夹拷贝到服务器即可。
# 依赖：PyInstaller（KRPC 环境已装）

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

Push-Location $root
try {
    pyinstaller `
        --name tofnet_generate `
        --onedir `
        --clean `
        --noconfirm `
        --paths . `
        --collect-all gfold `
        --hidden-import generation.generate `
        scripts\run_generate.py

    Write-Host ""
    Write-Host "完成。可执行文件：dist\tofnet_generate\tofnet_generate.exe"
    Write-Host "整个 dist\tofnet_generate\ 目录拷贝到服务器即可运行："
    Write-Host "  tofnet_generate.exe --n-samples 100000"
}
finally {
    Pop-Location
}

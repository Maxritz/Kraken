# ============================================================================
#  build_windows.ps1 — native Windows 11 build of kraken.
#
#  Requires the HIP SDK for Windows (ROCm/HIP SDK for Radeon and Ryzen).
#  Install it, then run this from a normal PowerShell prompt; the script
#  locates hipcc itself and falls back to a clear error if it cannot.
#
#    .\scripts\build_windows.ps1                 # auto-detect the GPU target
#    .\scripts\build_windows.ps1 -Targets gfx1201
#    .\scripts\build_windows.ps1 -Targets "gfx1031;gfx1201"   # fat binary
#    .\scripts\build_windows.ps1 -CpuOnly
#
#  Target cheat sheet (fleet priority: CPU oracle → gfx1031 → gfx1201):
#    gfx1031  Radeon RX 6700 XT   (RDNA2, no WMMA, packed math + v_dot2)  PRIMARY
#    gfx1030  Radeon RX 6800/6900 (RDNA2)
#    gfx1100  RX 7900 XTX         (RDNA3, WMMA 16x16x16 f16/bf16/i8)
#    gfx1201  RX 9070 XT          (RDNA4, WMMA gfx12 + fp8/bf16)          SECONDARY
#    gfx1200  RX 9070             (RDNA4)
# ============================================================================
[CmdletBinding()]
param(
    [string]$Targets = "",
    [switch]$CpuOnly,
    [string]$BuildDir = "build-hip",
    [int]$Jobs = 0
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root

function Find-HipSdk {
    $candidates = @()
    if ($env:HIP_PATH) { $candidates += $env:HIP_PATH }
    $candidates += @(
        "C:\Program Files\AMD\ROCm\*\bin",
        "C:\Program Files\AMD\HIP\*\bin",
        "$env:ProgramFiles\AMD\ROCm\*\bin"
    )
    foreach ($pattern in $candidates) {
        $found = Get-ChildItem -Path $pattern -Filter "hipcc.bat" -ErrorAction SilentlyContinue |
                 Select-Object -First 1
        if ($found) { return $found.FullName }
    }
    return $null
}

$hipcc = Find-HipSdk
if (-not $hipcc -and -not $CpuOnly) {
    Write-Host "kraken: HIP SDK not found." -ForegroundColor Yellow
    Write-Host "  Install the 'HIP SDK for Windows' from AMD, then re-run."
    Write-Host "  Building the CPU reference backend only."
    $CpuOnly = $true
}
if ($hipcc) {
    Write-Host "kraken: HIP SDK at $hipcc" -ForegroundColor Green
    $hipBin = Split-Path -Parent $hipcc
    if ($env:PATH -notlike "*$hipBin*") { $env:PATH = "$hipBin;$env:PATH" }
}

if (-not $Targets) {
    # Ask the driver for the first supported AMD device. rocminfo is not
    # always on PATH, so fall back to the two cards this engine targets.
    $Targets = "gfx1031;gfx1201"
    Write-Host "kraken: no -Targets given, building for $Targets" -ForegroundColor Yellow
}

$archFlags = ($Targets.Split(';') | ForEach-Object { "--offload-arch=$_" }) -join ' '
$cxxFlags = "-std=c++17 -O3 -ffast-math -fno-finite-math-only -D__HIP_PLATFORM_AMD__=1"

New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null
New-Item -ItemType Directory -Force -Path "$BuildDir\obj" | Out-Null

$coreSources = @(
    "src/common.cpp", "src/quant.cpp", "src/gguf.cpp", "src/tokenizer.cpp",
    "src/sampler.cpp", "src/model.cpp", "src/expert_cache.cpp", "src/engine.cpp",
    "src/backend_cpu.cpp", "src/json.cpp", "src/http.cpp", "src/server.cpp"
)

$objects = @()
$compiler = if ($CpuOnly) { "clang++" } else { $hipcc }

if ($CpuOnly) {
    Write-Host "kraken: CPU-only build (no HIP)" -ForegroundColor Yellow
} else {
    Write-Host "kraken: compiling HIP backend for [$Targets]" -ForegroundColor Green
}

foreach ($src in $coreSources) {
    $obj = "$BuildDir\obj\" + [IO.Path]::GetFileNameWithoutExtension($src) + ".obj"
    & $compiler $cxxFlags -Iinclude -c $src -o $obj
    if ($LASTEXITCODE -ne 0) { throw "compile failed: $src" }
    $objects += $obj
}

if ($CpuOnly) {
    & $compiler $cxxFlags -Iinclude -c src/hip/hip_stub.cpp -o "$BuildDir\obj\hip_stub.obj"
    if ($LASTEXITCODE -ne 0) { throw "compile failed: hip_stub.cpp" }
    $objects += "$BuildDir\obj\hip_stub.obj"
    & $compiler $cxxFlags -Iinclude -c src/main_cli.cpp -o "$BuildDir\obj\main_cli.obj"
    $objects += "$BuildDir\obj\main_cli.obj"
    & $compiler $cxxFlags -Iinclude $objects -o "$BuildDir\kraken.exe"
    & $compiler $cxxFlags -Iinclude -c src/main_server.cpp -o "$BuildDir\obj\main_server.obj"
    & $compiler $cxxFlags -Iinclude $objects "$BuildDir\obj\main_server.obj" `
        -o "$BuildDir\kraken-server.exe"
} else {
    & $hipcc $archFlags $cxxFlags -Iinclude -Isrc -DKRK_ENABLE_HIP=1 `
        -c src/hip/backend_hip.hip -o "$BuildDir\obj\backend_hip.obj"
    if ($LASTEXITCODE -ne 0) { throw "compile failed: backend_hip.hip" }
    $objects += "$BuildDir\obj\backend_hip.obj"
    & $hipcc $archFlags $cxxFlags -Iinclude -Isrc -DKRK_ENABLE_HIP=1 `
        -c src/main_cli.cpp -o "$BuildDir\obj\main_cli.obj"
    $objects += "$BuildDir\obj\main_cli.obj"
    & $hipcc $archFlags $cxxFlags $objects -o "$BuildDir\kraken.exe"
    if ($LASTEXITCODE -ne 0) { throw "link failed: kraken.exe" }
    & $hipcc $archFlags $cxxFlags -Iinclude -Isrc -DKRK_ENABLE_HIP=1 `
        -c src/main_server.cpp -o "$BuildDir\obj\main_server.obj"
    & $hipcc $archFlags $cxxFlags $objects "$BuildDir\obj\main_server.obj" `
        -o "$BuildDir\kraken-server.exe"
    if ($LASTEXITCODE -ne 0) { throw "link failed: kraken-server.exe" }
}

& $compiler $cxxFlags -Iinclude -c tools/inspect_gguf.cpp -o "$BuildDir\obj\inspect.obj"
& $compiler $cxxFlags -Iinclude -c tests/test_kraken.cpp -o "$BuildDir\obj\test.obj"
$coreObjs = $objects | Where-Object { $_ -notmatch "main_cli" -and $_ -notmatch "main_server" -and $_ -notmatch "backend_hip" }
$hipObj = @()
if (-not $CpuOnly) { $hipObj = @("$BuildDir\obj\backend_hip.obj") }

& $compiler $cxxFlags -Iinclude "$BuildDir\obj\inspect.obj" $coreObjs `
    -o "$BuildDir\kraken-inspect.exe"
& $compiler $cxxFlags -Iinclude "$BuildDir\obj\test.obj" $coreObjs $hipObj `
    -o "$BuildDir\kraken-tests.exe"

# ---------------------------------------------------------------------------
#  Stage the SDK's HIP runtime next to the binaries.
#
#  Windows resolves a DLL from the executable's own directory before System32,
#  and the driver installer also drops an amdhip64_<major>.dll there. Without
#  this step the binaries silently run whichever runtime the driver installed,
#  which may be older than the SDK they were compiled against (kraken prints
#  both versions and warns when they disagree).
# ---------------------------------------------------------------------------
if (-not $CpuOnly -and $hipcc) {
    $hipBin = Split-Path -Parent $hipcc
    $runtime = @()
    $runtime += Get-ChildItem "$hipBin\amdhip64_*.dll" -ErrorAction SilentlyContinue
    foreach ($name in @("amd_comgr.dll", "rocm_kpack.dll")) {
        $p = Join-Path $hipBin $name
        if (Test-Path $p) { $runtime += Get-Item $p }
    }
    foreach ($f in $runtime) {
        Copy-Item $f.FullName (Join-Path $BuildDir $f.Name) -Force
    }
    if ($runtime.Count -gt 0) {
        Write-Host "kraken: staged HIP runtime: $(($runtime | ForEach-Object { $_.Name }) -join ', ')" -ForegroundColor Green
    }
}

Write-Host ""
Write-Host "built:" -ForegroundColor Green
Get-ChildItem "$BuildDir\*.exe" | ForEach-Object { Write-Host "  $($_.Name)" }
Write-Host ""
Write-Host "run the tests:   $BuildDir\kraken-tests.exe"
Write-Host "inspect a model: $BuildDir\kraken-inspect.exe model.gguf"
Write-Host "generate:        $BuildDir\kraken.exe --model model.gguf --prompt `"Hello`""
Write-Host "serve OpenAI:    $BuildDir\kraken-server.exe --model model.gguf --port 8080"

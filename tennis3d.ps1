# Start Tennis3D from this source folder on Windows. Tennis3D.bat runs this file.
#
# The first run downloads a pinned uv, a private Python, the locked packages
# and the model weights into local-app\ beside this script, then opens the
# app. Later runs start offline. Nothing is installed or changed outside this
# folder, and deleting local-app\ removes everything this script created.
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$appHome = Join-Path $root 'local-app'
$runtime = Join-Path $appHome 'runtime'
$temporary = Join-Path $runtime 'tmp'
$uvVersion = '0.12.18'
$uvSha256 = 'cae6a3bc25239f83dffb467a4b180508d9da23986c04639ebfa44e43e6a84bff'
$pythonVersion = '3.12'

function Invoke-Checked {
  $command, $rest = $args
  & $command @rest
  if ($LASTEXITCODE -ne 0) { throw "Command failed: $($args -join ' ')" }
}

if (-not [Environment]::Is64BitOperatingSystem) { throw 'Tennis3D needs 64-bit Windows.' }
$profileName = $env:TENNIS3D_PROFILE
if (-not $profileName) {
  $profileName = 'cpu'
  if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    # Windows PowerShell turns redirected error output into a terminating
    # error under 'Stop', so a failing probe must select the CPU instead.
    $ErrorActionPreference = 'Continue'
    try {
      & nvidia-smi *> $null
      if ($LASTEXITCODE -eq 0) { $profileName = 'cu126' }
    }
    catch { $profileName = 'cpu' }
    finally { $ErrorActionPreference = 'Stop' }
  }
}
if ($profileName -eq 'intel') { throw 'The Intel profile supports Linux x86_64 only. Use cpu or cu126 on Windows.' }
if ($profileName -notin 'cpu', 'cu126') { throw 'TENNIS3D_PROFILE must be cpu or cu126.' }
$lock = Join-Path $root "tennis3d\resources\inference-$profileName.txt"
$buildLock = Join-Path $root 'tennis3d\resources\build.txt'
$project = Join-Path $root 'pyproject.toml'
foreach ($needed in $lock, $buildLock, $project) {
  if (-not (Test-Path -LiteralPath $needed)) { throw "$needed is missing. Download the whole repository and run Tennis3D.bat from it." }
}

# Only this folder's uv settings apply, so an ambient uv, Python or Tennis3D
# configuration cannot move the install elsewhere. These changes last for this
# process only.
Get-ChildItem Env: | Where-Object { $_.Name -like 'UV_*' } | ForEach-Object { Remove-Item -LiteralPath "Env:$($_.Name)" }
foreach ($name in 'PYTHONPATH', 'PYTHONHOME', 'PYTHONSTARTUP', 'PYTHONUSERBASE', 'VIRTUAL_ENV') {
  Remove-Item -LiteralPath "Env:$name" -ErrorAction SilentlyContinue
}
$env:TENNIS3D_HOME = $appHome
$env:UV_NO_CONFIG = '1'
$env:UV_MANAGED_PYTHON = '1'
$env:UV_PYTHON_INSTALL_BIN = '0'
$env:UV_PYTHON_INSTALL_REGISTRY = '0'
$env:UV_CACHE_DIR = Join-Path $runtime 'cache'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $runtime 'python'
$env:UV_PYTHON_BIN_DIR = Join-Path $runtime 'python-bin'
$env:UV_TOOL_DIR = Join-Path $runtime 'tools'
$env:UV_TOOL_BIN_DIR = Join-Path $runtime 'tools-bin'
$env:UV_CREDENTIALS_DIR = Join-Path $runtime 'credentials'
# Run this source tree only: no user site-packages or current-directory imports,
# with bytecode and framework caches kept in local-app\.
$env:PYTHONNOUSERSITE = '1'
$env:PYTHONSAFEPATH = '1'
$env:PYTHONPYCACHEPREFIX = Join-Path $runtime 'pycache'
$env:TORCH_HOME = Join-Path $runtime 'torch'
$env:PYTORCH_KERNEL_CACHE_PATH = Join-Path $runtime 'torch\kernels'
$env:CUDA_CACHE_PATH = Join-Path $runtime 'cuda'

$uv = Join-Path $runtime "uv-$uvVersion\uv.exe"
$python = Join-Path $runtime 'venv\Scripts\python.exe'
$stamp = Join-Path $runtime 'installed'
# The environment holds absolute paths, so a moved folder is installed again.
$identity = "$profileName uv-$uvVersion python-$pythonVersion $(($lock, $buildLock, $project | ForEach-Object { (Get-FileHash -LiteralPath $_ -Algorithm SHA256).Hash }) -join '-') $root"

function Get-Uv {
  $staging = Join-Path $temporary 'uv-download'
  if (Test-Path -LiteralPath $staging) { Remove-Item -LiteralPath $staging -Recurse -Force }
  New-Item -ItemType Directory -Force -Path $staging | Out-Null
  $archive = Join-Path $staging 'uv.zip'
  [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
  Invoke-WebRequest -UseBasicParsing -OutFile $archive `
    -Uri "https://github.com/astral-sh/uv/releases/download/$uvVersion/uv-x86_64-pc-windows-msvc.zip"
  if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash -ne $uvSha256) {
    throw "The downloaded uv $uvVersion does not match its pinned SHA-256."
  }
  $unpacked = Join-Path $staging 'uv'
  Expand-Archive -LiteralPath $archive -DestinationPath $unpacked
  $target = Split-Path -Parent $uv
  if (Test-Path -LiteralPath $target) { Remove-Item -LiteralPath $target -Recurse -Force }
  Move-Item -LiteralPath $unpacked -Destination $target
  Remove-Item -LiteralPath $staging -Recurse -Force
}

# Setup commands keep their temporary files in local-app\ too. The app keeps
# the user's own temporary folder, which the browser it opens inherits.
$userTemp = $env:TEMP
$userTmp = $env:TMP
$setupLock = $null
function Test-Installed {
  (Test-Path -LiteralPath $stamp) -and (Test-Path -LiteralPath $python) -and
    ((Get-Content -LiteralPath $stamp -Raw).Trim() -eq $identity)
}
try {
  New-Item -ItemType Directory -Force -Path $temporary | Out-Null
  $env:TEMP = $temporary
  $env:TMP = $temporary
  if (-not (Test-Installed)) {
    # One setup at a time: a second start would delete the environment the
    # first is building. Windows releases the file when its process ends.
    try { $setupLock = [IO.File]::Open((Join-Path $runtime 'setup.lock'), 'OpenOrCreate', 'ReadWrite', 'None') }
    catch { throw 'Another Tennis3D setup is running in this folder. Wait for it to finish, then start again.' }
  }
  # Another start may have finished the setup before this one took its turn.
  if ($setupLock -and -not (Test-Installed)) {
    Write-Host "Setting up Tennis3D ($profileName) in $appHome. This needs an internet connection once."
    # The stamp is written last, so an interrupted setup starts again here.
    if (Test-Path -LiteralPath $stamp) { Remove-Item -LiteralPath $stamp -Force }
    if (-not (Test-Path -LiteralPath $uv)) { Get-Uv }
    $venv = Join-Path $runtime 'venv'
    if (Test-Path -LiteralPath $venv) { Remove-Item -LiteralPath $venv -Recurse -Force }
    Invoke-Checked $uv venv --quiet --no-project --python $pythonVersion $venv
    Invoke-Checked $uv pip install --python $python --require-hashes `
      --extra-index-url "https://download.pytorch.org/whl/$profileName" --index-strategy unsafe-best-match `
      -r $lock
    # An editable install runs the package from this folder, so a source update
    # needs no reinstall. uv splits these paths at spaces, so they are relative.
    Push-Location -LiteralPath $root
    try { Invoke-Checked $uv pip install --python $python --no-deps --build-constraints tennis3d/resources/build.txt -e . }
    finally { Pop-Location }
    Invoke-Checked $python -m tennis3d version | Out-Null
    Set-Content -LiteralPath "$stamp.new" -Value $identity -Encoding UTF8
    Move-Item -LiteralPath "$stamp.new" -Destination $stamp -Force
  }
  if ($setupLock) {
    $setupLock.Close()
    $setupLock = $null
  }
  $ErrorActionPreference = 'Continue'
  # An explicit `setup` command chooses how to get the weights itself.
  if ($args.Count -eq 0 -or $args[0] -ne 'setup') {
    & $python -m tennis3d setup --check *> $null
    if ($LASTEXITCODE -ne 0) {
      Write-Host "Fetching the model weights into $appHome\weights."
      & $python -m tennis3d setup
      if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
  }
}
finally {
  if ($setupLock) { $setupLock.Close() }
  $env:TEMP = $userTemp
  $env:TMP = $userTmp
}

if ($args.Count -gt 0) { & $python -m tennis3d @args }
else { & $python -m tennis3d dashboard --gallery (Join-Path $root 'examples\gallery-v2\gallery.json') }
exit $LASTEXITCODE

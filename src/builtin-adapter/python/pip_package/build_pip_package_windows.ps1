# Copyright 2025 The AI Edge Model Explorer Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Stop script on first error
$ErrorActionPreference = 'Stop'

# Usage information
$Usage = @"
$(Split-Path -Leaf $MyInvocation.MyCommand.Path) <package-version>

Builds a pip package for the Model Explorer backend adapter for Windows.

<package-version> should be a string of the form "x.x.x", eg. "1.2.0".
"@

# Check for package version argument
if ($args.Count -lt 1) {
    Write-Output $Usage
    exit 1
}

$PackageVersion = $args[0]

# Define a regex pattern for the format x.x.x
$Pattern = "^\d+\.\d+\.\d+$"

# Check if the argument matches the pattern
if ($PackageVersion -notmatch $Pattern) {
    Write-Error "Error: The package version '$PackageVersion' is not in the correct format."
    exit 1
}

$env:PACKAGE_VERSION = $PackageVersion

# --- Script and Environment Setup ---
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$Python = if ($env:CI_BUILD_PYTHON) { $env:CI_BUILD_PYTHON } else { "python3" }
$PythonVersion = (& $Python --version).Split(' ')[1]
$VersionParts = $PythonVersion.Split('.')
# TF only supports python version ["3.9", "3.10", "3.11", "3.12"].
$env:TF_PYTHON_VERSION = "$($VersionParts[0]).$($VersionParts[1])"
$env:PROJECT_NAME = if ($env:WHEEL_PROJECT_NAME) { $env:WHEEL_PROJECT_NAME } else { "ai_edge_model_explorer_adapter" }
$BuildDir = Join-Path $PSScriptRoot "gen/adapter_pip"

# --- Build Process for Windows ---

# Build source tree
if (Test-Path $BuildDir) {
    Remove-Item -Path $BuildDir -Recurse -Force
}
New-Item -Path (Join-Path $BuildDir "ai_edge_model_explorer_adapter") -ItemType Directory -Force | Out-Null

Copy-Item -Path (Join-Path $ScriptDir "MANIFEST.in") -Destination $BuildDir
Copy-Item -Path (Join-Path $ScriptDir "setup_with_binary.py") -Destination (Join-Path $BuildDir "setup.py")
Set-Content -Path (Join-Path $BuildDir "ai_edge_model_explorer_adapter/__init__.py") -Value "__version__ = '$($env:PACKAGE_VERSION)'"

# Build python _pywrap_convert_wrapper for Windows
# We need to pass down the environment variable with a possible alternate Python
# include path for Python 3.x builds to work.
$env:CROSSTOOL_PYTHON_INCLUDE_PATH = $env:CROSSTOOL_PYTHON_INCLUDE_PATH
$LibraryExtension = ".pyd"
# The .bazelrc handles all compiler flags. We just need to trigger the config.
$BazelPlatformFlags = "--config=windows"

# Note: --config=monolithic is already set for Windows in .bazelrc.
Write-Host "Starting Bazel build for Windows..."
bazel build -c opt -s --config=monolithic --config=noaws --config=nogcp --config=nohdfs --config=nonccl $BazelPlatformFlags python/convert_wrapper:_pywrap_convert_wrapper

$WrapperSourcePath = "bazel-bin/python/convert_wrapper/_pywrap_convert_wrapper$($LibraryExtension)"
$WrapperDestPath = Join-Path $BuildDir "ai_edge_model_explorer_adapter"
Copy-Item -Path $WrapperSourcePath -Destination $WrapperDestPath

# Bazel can generate the wrapper library as read-only.
# We need write permissions for setuptools to clean up the build directory.
$FinalWrapperPath = Join-Path $WrapperDestPath "_pywrap_convert_wrapper$($LibraryExtension)"
Set-ItemProperty -Path $FinalWrapperPath -Name IsReadOnly -Value $false

# Build python wheel for Windows.
Set-Location $BuildDir
$WheelPlatformName = "win_amd64"

# Execute the setup script
& $Python setup.py sdist bdist_wheel --plat-name=$WheelPlatformName

Write-Host "Output can be found here:"
Get-ChildItem -Path (Get-Location) -Filter "*.whl" -Recurse | ForEach-Object { $_.FullName }

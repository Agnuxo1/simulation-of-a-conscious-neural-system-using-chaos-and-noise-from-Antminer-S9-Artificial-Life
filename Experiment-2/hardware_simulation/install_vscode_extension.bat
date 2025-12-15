@echo off
echo Installing Verilog-HDL/SystemVerilog VS Code Extension
echo.
echo This extension provides syntax highlighting, linting, and other features for Verilog and SystemVerilog files.
echo.

echo Method 1: Using VS Code GUI
echo =========================
echo 1. Open VS Code
echo 2. Click on the Extensions icon (or Ctrl+Shift+X)
echo 3. Search for "Verilog-HDL/SystemVerilog" by mshr-h
echo 4. Click Install
echo.

echo Method 2: Using Command Line (if VS Code is in PATH)
echo ==================================================
echo code --install-extension mshr-h.verilog-hdl-systemverilog
echo.

echo Method 3: Manual Installation
echo =============================
echo 1. Download the VSIX file from:
echo    https://marketplace.visualstudio.com/items?itemName=mshr-h.verilog-hdl-systemverilog
echo 2. Open VS Code
echo 3. Press Ctrl+Shift+P to open Command Palette
echo 4. Type "Extensions: Install from VSIX"
echo 5. Select the downloaded VSIX file
echo.

echo Additional Recommended Extensions:
echo ==================================
echo - "iverilog" by meganeredd (iverilog syntax support)
echo - "GTKWave" by asu-neu (GTKWave integration)
echo - "Verilog Testbench Runner" by asu-neu
echo.

echo Verification:
echo =============
echo After installation:
echo 1. Open a .v or .sv file in VS Code
echo 2. You should see syntax highlighting
echo 3. Check the Output panel for any linting errors
echo.

pause
@echo off
echo Installing Verilator for Windows
echo.
echo Verilator is a free Verilog HDL simulator and compiler for SystemVerilog.
echo This script will guide you through the manual installation process.
echo.

echo Step 1: Download Verilator
echo ========================
echo Visit: https://www.veripool.org/verilator
echo Download the latest Windows binary release (usually verilator-X.X.X-win64.zip)
echo.

echo Step 2: Extract the archive
echo ==========================
echo Extract the downloaded ZIP file to a permanent location like:
echo C:\Tools\verilator\
echo.

echo Step 3: Add to PATH
echo =================
echo Add C:\Tools\verilator\bin to your system PATH environment variable.
echo You can do this through:
echo - Control Panel ^> System ^> Advanced System Settings ^> Environment Variables
echo - Or use: set PATH=%PATH%;C:\Tools\verilator\bin
echo.

echo Step 4: Verify Installation
echo =========================
echo After adding to PATH, open a new command prompt and run:
echo verilator --version
echo.

echo Step 5: Install dependencies (optional)
echo =====================================
echo For full functionality, you may also need:
echo - Perl (for testbench generation)
echo - Make (usually included with Git for Windows)
echo.

pause
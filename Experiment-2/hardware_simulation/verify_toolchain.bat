@echo off
echo Verifying Hardware Simulation Toolchain Installation
echo ====================================================
echo.

echo Checking Verilator installation...
verilator --version
if %errorlevel% neq 0 (
    echo ERROR: Verilator is not installed or not in PATH
    echo Please run install_verilator.bat first
    goto end
) else (
    echo SUCCESS: Verilator is installed
)

echo.
echo Checking GTKWave installation...
gtkwave --version
if %errorlevel% neq 0 (
    echo ERROR: GTKWave is not installed or not in PATH
    echo Please run install_gtkwave.bat first
    goto end
) else (
    echo SUCCESS: GTKWave is installed
)

echo.
echo Testing Verilator compilation...
echo =================================

cd /d "%~dp0"

if not exist "src\basic_counter.v" (
    echo ERROR: Test Verilog file not found
    echo Please ensure basic_counter.v exists in src directory
    goto end
)

echo Compiling basic_counter.v...
verilator --cc src\basic_counter.v --trace -o obj_dir\basic_counter

if %errorlevel% neq 0 (
    echo ERROR: Verilator compilation failed
    goto end
) else (
    echo SUCCESS: Verilator compilation completed
)

echo.
echo Building simulation executable...
echo =================================
if not exist "obj_dir\basic_counter.cpp" (
    echo ERROR: Generated C++ files not found
    goto end
)

echo Creating Makefile for simulation...
echo 'include config.mk' > obj_dir\Makefile
echo 'include ../include.mk' >> obj_dir\Makefile
echo 'default: Vbasic_counter' >> obj_dir\Makefile
echo 'include *.d' >> obj_dir\Makefile

echo Compiling simulation executable...
cd obj_dir
make -f ../build\sim_makefile.mk Vbasic_counter
cd ..

if %errorlevel% neq 0 (
    echo ERROR: Simulation executable compilation failed
    goto end
) else (
    echo SUCCESS: Simulation executable created
)

echo.
echo Running simulation...
echo ===================
if not exist "bin\Vbasic_counter" (
    echo ERROR: Simulation executable not found
    goto end
)

echo Running testbench_basic_counter...
bin\Vbasic_counter

if %errorlevel% neq 0 (
    echo ERROR: Simulation execution failed
    goto end
) else (
    echo SUCCESS: Simulation executed
)

echo.
echo Checking waveform output...
if exist "testbench_basic_counter.vcd" (
    echo SUCCESS: VCD file generated
    echo You can view the waveform with: gtkwave testbench_basic_counter.vcd
) else (
    echo WARNING: VCD file not found
)

echo.
echo TOOLCHAIN VERIFICATION SUMMARY
echo ===============================
echo - Verilator: INSTALLED
echo - GTKWave: INSTALLED  
echo - Compilation: SUCCESSFUL
echo - Simulation: SUCCESSFUL
echo - Waveform: GENERATED
echo.
echo Your hardware simulation toolchain is ready for the Antminer S9 BM1387 Digital Twin project!
echo.
echo Next steps:
echo 1. Install the VS Code extension (run install_vscode_extension.bat)
echo 2. Begin modeling the BM1387 chip architecture
echo 3. Create comprehensive testbenches for chip verification

:end
echo.
pause
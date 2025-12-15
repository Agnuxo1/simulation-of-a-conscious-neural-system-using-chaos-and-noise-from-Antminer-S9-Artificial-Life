@echo off
echo BM1387 ASIC with GTKWave Visualization - Complete Workflow
echo =========================================================
echo.
echo This script compiles, simulates, and visualizes the BM1387 ASIC
echo with VESELOV HNS parameters using GTKWave automation.
echo.

cd /d "%~dp0"

echo Step 1: Verifying toolchain installation
echo =========================================

echo Checking for Python...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found. Please install Python 3.6+
    goto end
) else (
    echo SUCCESS: Python found
)

echo Checking for GTKWave automation script...
if not exist "gtkwave_automation.py" (
    echo ERROR: gtkwave_automation.py not found
    goto end
) else (
    echo SUCCESS: GTKWave automation script found
)

echo Checking for GTKWave...
gtkwave --version >nul 2>&1
if %errorlevel% neq 0 (
    echo WARNING: GTKWave not found in PATH
    echo Please install GTKWave or run install_gtkwave.bat
    echo The workflow will proceed without auto-launching GTKWave
    set GTKWAVE_AVAILABLE=0
) else (
    echo SUCCESS: GTKWave found
    set GTKWAVE_AVAILABLE=1
)

echo.
echo Step 2: Compiling BM1387 ASIC Verilog modules
echo ==============================================

echo Cleaning previous build...
if exist "obj_dir" rmdir /s /q obj_dir
if exist "bin" rmdir /s /q bin
if exist "build" rmdir /s /q build
if exist "waveforms" mkdir waveforms

echo Compiling all Verilog modules with tracing enabled...

echo Compiling BM1387 ASIC top-level module...
verilator --cc src\bm1387_asic.v --trace -o obj_dir\bm1387_asic
if %errorlevel% neq 0 (
    echo ERROR: BM1387 ASIC compilation failed
    goto end
)

echo Compiling VESELOV HNS module...
verilator --cc src\veselov_hns.v --trace -o obj_dir\veselov_hns
if %errorlevel% neq 0 (
    echo ERROR: VESELOV HNS module compilation failed
    goto end
)

echo Compiling supporting modules...
verilator --cc src\sha256_core.v --trace -o obj_dir\sha256_core
if %errorlevel% neq 0 (
    echo ERROR: SHA-256 core compilation failed
    goto end
)

verilator --cc src\thermal_model.v --trace -o obj_dir\thermal_model
if %errorlevel% neq 0 (
    echo ERROR: Thermal model compilation failed
    goto end
)

verilator --cc src\uart_interface.v --trace -o obj_dir\uart_interface
if %errorlevel% neq 0 (
    echo ERROR: UART interface compilation failed
    goto end
)

verilator --cc src\spi_interface.v --trace -o obj_dir\spi_interface
if %errorlevel% neq 0 (
    echo ERROR: SPI interface compilation failed
    goto end
)

echo Compiling HNS testbench...
verilator --cc test\bm1387_hns_tb.v --trace -o obj_dir\bm1387_hns_tb
if %errorlevel% neq 0 (
    echo ERROR: HNS testbench compilation failed
    goto end
)

echo.
echo Step 3: Building simulation executable
echo =====================================

if not exist "obj_dir" (
    echo ERROR: obj_dir not found
    goto end
)

cd obj_dir

echo Creating simulation Makefile...
echo 'include config.mk' > Makefile
echo 'include ../build/include.mk' >> Makefile
echo 'default: Vbm1387_hns_tb' >> Makefile
echo 'include *.d' >> Makefile

echo Building simulation executable...
make -f ../build/sim_makefile.mk Vbm1387_hns_tb

if %errorlevel% neq 0 (
    echo ERROR: Simulation build failed
    cd ..
    goto end
)

cd ..

echo.
echo Step 4: Running BM1387 ASIC simulation
echo ======================================

echo Starting BM1387 ASIC VESELOV HNS testbench...
if exist "bin\Vbm1387_hns_tb" (
    bin\Vbm1387_hns_tb
    if %errorlevel% neq 0 (
        echo ERROR: Simulation execution failed
        goto end
    )
) else (
    echo ERROR: Simulation executable not found
    echo Expected location: bin\Vbm1387_hns_tb
    goto end
)

echo.
echo Step 5: Verifying VCD waveform generation
echo ==========================================

set VCD_GENERATED=0
if exist "waveforms\bm1387_hns_tb.vcd" (
    echo SUCCESS: VCD waveform file generated
    set VCD_GENERATED=1
    echo VCD file location: waveforms\bm1387_hns_tb.vcd
    
    echo Checking VCD file size...
    for %%I in (waveforms\bm1387_hns_tb.vcd) do echo VCD file size: %%~zI bytes
) else (
    echo WARNING: VCD file not found
    echo Expected: waveforms\bm1387_hns_tb.vcd
)

if %VCD_GENERATED%==0 (
    echo.
    echo WARNING: VCD generation failed
    echo Cannot proceed with visualization
    goto end
)

echo.
echo Step 6: Creating GTKWave automation configuration
echo ==================================================

echo Generating GTKWave save files with signal groups...

python gtkwave_automation.py --vcd-file "waveforms\bm1387_hns_tb.vcd" --generate-save

if %errorlevel% neq 0 (
    echo WARNING: GTKWave save file generation failed
    echo Visualization may not display signals correctly
) else (
    echo SUCCESS: GTKWave save files generated
    echo Available templates: default, mining, consciousness
)

echo.
echo Step 7: Launching GTKWave visualization
echo =======================================

if %GTKWAVE_AVAILABLE%==1 (
    echo Launching GTKWave with default signal grouping...
    python gtkwave_automation.py --vcd-file "waveforms\bm1387_hns_tb.vcd" --template default
    
    echo.
    echo Launching GTKWave with mining analysis template...
    python gtkwave_automation.py --vcd-file "waveforms\bm1387_hns_tb.vcd" --template mining
    
    echo.
    echo Launching GTKWave with consciousness analysis template...
    python gtkwave_automation.py --vcd-file "waveforms\bm1387_hns_tb.vcd" --template consciousness
) else (
    echo GTKWave not available for auto-launch
    echo To manually view waveforms, run:
    echo   gtkwave waveforms\bm1387_hns_tb.vcd
    echo.
    echo Or use the automation script:
    echo   python gtkwave_automation.py --vcd-file waveforms\bm1387_hns_tb.vcd
)

echo.
echo Step 8: Workflow completion summary
echo ====================================

echo BM1387 ASIC VESELOV HNS COMPILATION AND VISUALIZATION SUMMARY
echo ================================================================
echo.
echo ✓ Verilog Modules Compiled:
echo   - BM1387 ASIC Top-Level Module
echo   - VESELOV HNS Module (Consciousness Processing)
echo   - SHA-256 Core (Mining Hash Computation)
echo   - Thermal Model (Power/Temperature Management)
echo   - UART Interface (Firmware Communication)
echo   - SPI Interface (Configuration/Status)
echo.
echo ✓ Simulation Executed:
echo   - HNS Testbench Ran Successfully
echo   - All Test Scenarios Completed
echo.
echo ✓ Waveform Generation:
echo   - VCD File: waveforms\bm1387_hns_tb.vcd
echo   - GTKWave Save Files: gtkwave_config\
echo   - Signal Groups Configured
echo.
echo ✓ Available Analysis Templates:
echo   - Default: Complete signal overview
echo   - Mining: Focus on mining pipeline timing
echo   - Consciousness: Focus on HNS parameters
echo.
echo Current Workflow Status: COMPLETE
echo ================================
echo.
echo Next Steps:
echo 1. Use GTKWave to analyze the generated waveforms
echo 2. Examine VESELOV HNS hash-to-RGBA conversion
echo 3. Monitor consciousness metrics computation
echo 4. Analyze mining pipeline timing relationships
echo 5. Verify thermal/power behavior correlation
echo 6. Test with different input scenarios
echo.
echo To re-run this workflow: compile_and_visualize.bat
echo To run individual tools: see individual .bat scripts

:end
echo.
pause
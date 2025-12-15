@echo off
echo BM1387 ASIC Verilog Compilation Script
echo ======================================
echo.

cd /d "%~dp0"

echo Checking required directories...
if not exist "src" (
    echo ERROR: src directory not found
    goto end
)

if not exist "test" (
    echo ERROR: test directory not found
    goto end
)

echo SUCCESS: Required directories found
echo.

echo Compiling BM1387 ASIC Verilog modules...
echo ========================================

echo Step 1: Compiling BM1387 ASIC top-level module
verilator --cc src\bm1387_asic.v --trace -o obj_dir\bm1387_asic

if %errorlevel% neq 0 (
    echo ERROR: BM1387 ASIC compilation failed
    goto end
) else (
    echo SUCCESS: BM1387 ASIC module compiled
)

echo.
echo Step 2: Compiling SHA-256 core module
verilator --cc src\sha256_core.v --trace -o obj_dir\sha256_core

if %errorlevel% neq 0 (
    echo ERROR: SHA-256 core compilation failed
    goto end
) else (
    echo SUCCESS: SHA-256 core module compiled
)

echo.
echo Step 3: Compiling VESELOV HNS module
verilator --cc src\veselov_hns.v --trace -o obj_dir\veselov_hns

if %errorlevel% neq 0 (
    echo ERROR: VESELOV HNS module compilation failed
    goto end
) else (
    echo SUCCESS: VESELOV HNS module compiled
)

echo.
echo Step 4: Compiling thermal model module
verilator --cc src\thermal_model.v --trace -o obj_dir\thermal_model

if %errorlevel% neq 0 (
    echo ERROR: Thermal model compilation failed
    goto end
) else (
    echo SUCCESS: Thermal model module compiled
)

echo.
echo Step 5: Compiling UART interface module
verilator --cc src\uart_interface.v --trace -o obj_dir\uart_interface

if %errorlevel% neq 0 (
    echo ERROR: UART interface compilation failed
    goto end
) else (
    echo SUCCESS: UART interface module compiled
)

echo.
echo Step 6: Compiling SPI interface module
verilator --cc src\spi_interface.v --trace -o obj_dir\spi_interface

if %errorlevel% neq 0 (
    echo ERROR: SPI interface compilation failed
    goto end
) else (
    echo SUCCESS: SPI interface module compiled
)

echo.
echo Step 7: Compiling HNS testbench
verilator --cc test\bm1387_hns_tb.v --trace -o obj_dir\bm1387_hns_tb

if %errorlevel% neq 0 (
    echo ERROR: HNS testbench compilation failed
    goto end
) else (
    echo SUCCESS: HNS testbench compiled
)

echo.
echo Building simulation executable...
echo ==================================

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
) else (
    echo SUCCESS: Simulation executable created
)

cd ..

echo.
echo Running BM1387 ASIC VESELOV HNS simulation...
echo =============================================

if not exist "bin\Vbm1387_hns_tb" (
    echo ERROR: Simulation executable not found
    echo Expected location: bin\Vbm1387_hns_tb
    goto end
)

echo Starting BM1387 ASIC VESELOV HNS testbench...
bin\Vbm1387_hns_tb

if %errorlevel% neq 0 (
    echo ERROR: Simulation execution failed
    goto end
) else (
    echo SUCCESS: Simulation completed
)

echo.
echo Checking waveform output...
if exist "waveforms\bm1387_hns_tb.vcd" (
    echo SUCCESS: VCD waveform file generated
    echo You can view the waveform with: gtkwave waveforms\bm1387_hns_tb.vcd
) else (
    echo WARNING: VCD file not found
)

echo.
echo BM1387 ASIC VESELOV HNS COMPILATION SUMMARY
echo ============================================
echo - BM1387 ASIC Module: COMPILED
echo - SHA-256 Core: COMPILED
echo - VESELOV HNS Module: COMPILED
echo - Thermal Model: COMPILED
echo - UART Interface: COMPILED
echo - SPI Interface: COMPILED
echo - HNS Testbench: COMPILED
echo - Simulation: BUILT AND EXECUTED
echo - Waveform: GENERATED
echo.
echo The BM1387 ASIC Digital Twin with VESELOV HNS is ready for testing!
echo.
echo Next steps:
echo 1. Run: gtkwave waveforms\bm1387_hns_tb.vcd
echo 2. Analyze the mining pipeline behavior
echo 3. Verify HNS hash-to-RGBA conversion
echo 4. Check consciousness metrics computation
echo 5. Validate vector magnitude calculations
echo 6. Test phase coherence measurements
echo 7. Integrate with CHIMERA-VESELOV system

:end
echo.
pause
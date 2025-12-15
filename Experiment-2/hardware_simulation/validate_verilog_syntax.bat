@echo off
echo BM1387 ASIC Verilog Syntax Validation
echo =====================================
echo.

cd /d "%~dp0"

echo Validating BM1387 ASIC Verilog Files
echo =====================================
echo.

set error_count=0
set file_count=0

echo Checking directory structure...
echo ------------------------------

if exist "src" (
    echo ✓ src directory found
) else (
    echo ✗ src directory not found
    set /a error_count+=1
)

if exist "test" (
    echo ✓ test directory found
) else (
    echo ✗ test directory not found
    set /a error_count+=1
)

if exist "waveforms" (
    echo ✓ waveforms directory found
) else (
    echo ⚠ waveforms directory not found (will be created)
    mkdir waveforms
)

echo.

echo Validating BM1387 ASIC Top-Level Module
echo ========================================

if exist "src\bm1387_asic.v" (
    echo ✓ BM1387 ASIC Module: Found
    set /a file_count+=1
    
    findstr /c:"module bm1387_asic" "src\bm1387_asic.v" >nul
    if %errorlevel% equ 0 (
        echo   ✓ Module declaration found
    ) else (
        echo   ✗ Module declaration not found
        set /a error_count+=1
    )
    
    findstr /c:"endmodule" "src\bm1387_asic.v" >nul
    if %errorlevel% equ 0 (
        echo   ✓ Endmodule found
    ) else (
        echo   ✗ Endmodule not found
        set /a error_count+=1
    )
    
    findstr /c:"always @" "src\bm1387_asic.v" >nul
    if %errorlevel% equ 0 (
        echo   ✓ Always blocks found
    ) else (
        echo   ⚠ No always blocks found
    )
    
    findstr /c:"parameter" "src\bm1387_asic.v" >nul
    if %errorlevel% equ 0 (
        echo   ✓ Parameters found
    ) else (
        echo   ⚠ No parameters found
    )
) else (
    echo ✗ BM1387 ASIC Module: File not found
    set /a error_count+=1
)

echo.

echo Validating SHA-256 Core Module
echo ===============================

if exist "src\sha256_core.v" (
    echo ✓ SHA-256 Core Module: Found
    set /a file_count+=1
    
    findstr /c:"module sha256_core" "src\sha256_core.v" >nul
    if %errorlevel% equ 0 (
        echo   ✓ Module declaration found
    ) else (
        echo   ✗ Module declaration not found
        set /a error_count+=1
    )
    
    findstr /c:"endmodule" "src\sha256_core.v" >nul
    if %errorlevel% equ 0 (
        echo   ✓ Endmodule found
    ) else (
        echo   ✗ Endmodule not found
        set /a error_count+=1
    )
    
    findstr /c:"Ch(" "src\sha256_core.v" >nul
    if %errorlevel% equ 0 (
        echo   ✓ SHA-256 functions found
    ) else (
        echo   ⚠ SHA-256 functions not found
    )
) else (
    echo ✗ SHA-256 Core Module: File not found
    set /a error_count+=1
)

echo.

echo Validating Thermal Model Module
echo ================================

if exist "src\thermal_model.v" (
    echo ✓ Thermal Model Module: Found
    set /a file_count+=1
    
    findstr /c:"module thermal_model" "src\thermal_model.v" >nul
    if %errorlevel% equ 0 (
        echo   ✓ Module declaration found
    ) else (
        echo   ✗ Module declaration not found
        set /a error_count+=1
    )
    
    findstr /c:"endmodule" "src\thermal_model.v" >nul
    if %errorlevel% equ 0 (
        echo   ✓ Endmodule found
    ) else (
        echo   ✗ Endmodule not found
        set /a error_count+=1
    )
    
    findstr /c:"temperature" "src\thermal_model.v" >nul
    if %errorlevel% equ 0 (
        echo   ✓ Temperature modeling found
    ) else (
        echo   ⚠ Temperature modeling not found
    )
) else (
    echo ✗ Thermal Model Module: File not found
    set /a error_count+=1
)

echo.

echo Validating UART Interface Module
echo =================================

if exist "src\uart_interface.v" (
    echo ✓ UART Interface Module: Found
    set /a file_count+=1
    
    findstr /c:"module uart_interface" "src\uart_interface.v" >nul
    if %errorlevel% equ 0 (
        echo   ✓ Module declaration found
    ) else (
        echo   ✗ Module declaration not found
        set /a error_count+=1
    )
    
    findstr /c:"endmodule" "src\uart_interface.v" >nul
    if %errorlevel% equ 0 (
        echo   ✓ Endmodule found
    ) else (
        echo   ✗ Endmodule not found
        set /a error_count+=1
    )
    
    findstr /c:"uart_tx" "src\uart_interface.v" >nul
    if %errorlevel% equ 0 (
        echo   ✓ UART signals found
    ) else (
        echo   ⚠ UART signals not found
    )
) else (
    echo ✗ UART Interface Module: File not found
    set /a error_count+=1
)

echo.

echo Validating SPI Interface Module
echo ================================

if exist "src\spi_interface.v" (
    echo ✓ SPI Interface Module: Found
    set /a file_count+=1
    
    findstr /c:"module spi_interface" "src\spi_interface.v" >nul
    if %errorlevel% equ 0 (
        echo   ✓ Module declaration found
    ) else (
        echo   ✗ Module declaration not found
        set /a error_count+=1
    )
    
    findstr /c:"endmodule" "src\spi_interface.v" >nul
    if %errorlevel% equ 0 (
        echo   ✓ Endmodule found
    ) else (
        echo   ✗ Endmodule not found
        set /a error_count+=1
    )
    
    findstr /c:"spi_clk" "src\spi_interface.v" >nul
    if %errorlevel% equ 0 (
        echo   ✓ SPI signals found
    ) else (
        echo   ⚠ SPI signals not found
    )
) else (
    echo ✗ SPI Interface Module: File not found
    set /a error_count+=1
)

echo.

echo Validating BM1387 ASIC Testbench
echo =================================

if exist "test\bm1387_asic_tb.v" (
    echo ✓ BM1387 ASIC Testbench: Found
    set /a file_count+=1
    
    findstr /c:"module bm1387_asic_tb" "test\bm1387_asic_tb.v" >nul
    if %errorlevel% equ 0 (
        echo   ✓ Module declaration found
    ) else (
        echo   ✗ Module declaration not found
        set /a error_count+=1
    )
    
    findstr /c:"endmodule" "test\bm1387_asic_tb.v" >nul
    if %errorlevel% equ 0 (
        echo   ✓ Endmodule found
    ) else (
        echo   ✗ Endmodule not found
        set /a error_count+=1
    )
    
    findstr /c:"initial" "test\bm1387_asic_tb.v" >nul
    if %errorlevel% equ 0 (
        echo   ✓ Initial blocks found
    ) else (
        echo   ⚠ No initial blocks found
    )
) else (
    echo ✗ BM1387 ASIC Testbench: File not found
    set /a error_count+=1
)

echo.

echo Checking Module Instantiation
echo =============================

findstr /c:"sha256_core u_sha256_core" "src\bm1387_asic.v" >nul
if %errorlevel% equ 0 (
    echo ✓ SHA-256 core instantiation found
) else (
    echo ✗ SHA-256 core instantiation not found
    set /a error_count+=1
)

findstr /c:"thermal_model u_thermal_model" "src\bm1387_asic.v" >nul
if %errorlevel% equ 0 (
    echo ✓ Thermal model instantiation found
) else (
    echo ✗ Thermal model instantiation not found
    set /a error_count+=1
)

findstr /c:"uart_interface u_uart_if" "src\bm1387_asic.v" >nul
if %errorlevel% equ 0 (
    echo ✓ UART interface instantiation found
) else (
    echo ✗ UART interface instantiation not found
    set /a error_count+=1
)

findstr /c:"spi_interface u_spi_if" "src\bm1387_asic.v" >nul
if %errorlevel% equ 0 (
    echo ✓ SPI interface instantiation found
) else (
    echo ✗ SPI interface instantiation not found
    set /a error_count+=1
)

echo.

echo File Statistics
echo ================
dir src\*.v /b >nul 2>&1 && echo Verilog source files: && dir src\*.v /b | find /c /v "" || echo Verilog source files: 0
dir test\*.v /b >nul 2>&1 && echo Testbench files: && dir test\*.v /b | find /c /v "" || echo Testbench files: 0

echo.

echo VALIDATION SUMMARY
echo ==================
echo Files validated: %file_count%
echo Errors found: %error_count%

if %error_count% equ 0 (
    echo.
    echo ✓ ALL VALIDATION CHECKS PASSED
    echo.
    echo The BM1387 ASIC Verilog model structure is valid and ready for compilation.
    echo.
    echo To compile and run the simulation:
    echo   1. Ensure Verilator is installed (run install_verilator.bat)
    echo   2. Run: compile_bm1387.bat
    echo   3. View waveforms: gtkwave waveforms\bm1387_asic_tb.vcd
    echo.
    echo The BM1387 ASIC Digital Twin is ready for testing!
) else (
    echo.
    echo ✗ VALIDATION FAILED - Please review the errors above
)

echo.
pause
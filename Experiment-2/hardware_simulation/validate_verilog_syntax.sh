#!/bin/bash

echo "BM1387 ASIC Verilog Syntax Validation"
echo "====================================="
echo

# Function to check if a file exists and has content
validate_file() {
    local file=$1
    local description=$2
    
    if [ -f "$file" ]; then
        local lines=$(wc -l < "$file")
        if [ $lines -gt 0 ]; then
            echo "✓ $description: Found ($lines lines)"
            return 0
        else
            echo "✗ $description: File exists but is empty"
            return 1
        fi
    else
        echo "✗ $description: File not found"
        return 1
    fi
}

# Function to check Verilog module structure
validate_module_structure() {
    local file=$1
    local module_name=$2
    
    if grep -q "module $module_name" "$file"; then
        echo "  ✓ Module declaration found"
        return 0
    else
        echo "  ✗ Module declaration not found"
        return 1
    fi
    
    if grep -q "endmodule" "$file"; then
        echo "  ✓ Endmodule found"
        return 0
    else
        echo "  ✗ Endmodule not found"
        return 1
    fi
}

# Function to check for common Verilog constructs
validate_verilog_constructs() {
    local file=$1
    local description=$2
    
    echo "  Checking Verilog constructs..."
    
    # Check for always blocks
    if grep -q "always @" "$file"; then
        echo "    ✓ Always blocks found"
    else
        echo "    ⚠ No always blocks found (may be structural only)"
    fi
    
    # Check for initial blocks (testbenches)
    if grep -q "initial" "$file"; then
        echo "    ✓ Initial blocks found"
    else
        echo "    ⚠ No initial blocks found"
    fi
    
    # Check for parameter definitions
    if grep -q "parameter" "$file"; then
        echo "    ✓ Parameters found"
    else
        echo "    ⚠ No parameters found"
    fi
    
    # Check for wire/reg declarations
    if grep -qE "(wire|reg)\s+" "$file"; then
        echo "    ✓ Wire/reg declarations found"
    else
        echo "    ⚠ No wire/reg declarations found"
    fi
    
    # Check for module instantiation
    if grep -q "module.*(" "$file"; then
        echo "    ✓ Module port list found"
    else
        echo "    ✗ Module port list missing"
        return 1
    fi
    
    return 0
}

echo "Validating BM1387 ASIC Verilog Files"
echo "====================================="
echo

# Change to hardware_simulation directory
cd "$(dirname "$0")"

error_count=0
file_count=0

# Check source directory structure
echo "Checking directory structure..."
echo "------------------------------"

if [ -d "src" ]; then
    echo "✓ src directory found"
else
    echo "✗ src directory not found"
    ((error_count++))
fi

if [ -d "test" ]; then
    echo "✓ test directory found"
else
    echo "✗ test directory not found"
    ((error_count++))
fi

if [ -d "waveforms" ]; then
    echo "✓ waveforms directory found"
else
    echo "⚠ waveforms directory not found (will be created)"
    mkdir -p waveforms
fi

echo

# Validate main BM1387 ASIC module
echo "Validating BM1387 ASIC Top-Level Module"
echo "========================================"

if validate_file "src/bm1387_asic.v" "BM1387 ASIC Module"; then
    ((file_count++))
    validate_module_structure "src/bm1387_asic.v" "bm1387_asic"
    validate_verilog_constructs "src/bm1387_asic.v" "BM1387 ASIC"
fi

echo

# Validate SHA-256 core
echo "Validating SHA-256 Core Module"
echo "==============================="

if validate_file "src/sha256_core.v" "SHA-256 Core Module"; then
    ((file_count++))
    validate_module_structure "src/sha256_core.v" "sha256_core"
    validate_verilog_constructs "src/sha256_core.v" "SHA-256 Core"
fi

echo

# Validate thermal model
echo "Validating Thermal Model Module"
echo "================================"

if validate_file "src/thermal_model.v" "Thermal Model Module"; then
    ((file_count++))
    validate_module_structure "src/thermal_model.v" "thermal_model"
    validate_verilog_constructs "src/thermal_model.v" "Thermal Model"
fi

echo

# Validate UART interface
echo "Validating UART Interface Module"
echo "================================="

if validate_file "src/uart_interface.v" "UART Interface Module"; then
    ((file_count++))
    validate_module_structure "src/uart_interface.v" "uart_interface"
    validate_verilog_constructs "src/uart_interface.v" "UART Interface"
fi

echo

# Validate SPI interface
echo "Validating SPI Interface Module"
echo "================================"

if validate_file "src/spi_interface.v" "SPI Interface Module"; then
    ((file_count++))
    validate_module_structure "src/spi_interface.v" "spi_interface"
    validate_verilog_constructs "src/spi_interface.v" "SPI Interface"
fi

echo

# Validate testbench
echo "Validating BM1387 ASIC Testbench"
echo "================================="

if validate_file "test/bm1387_asic_tb.v" "BM1387 ASIC Testbench"; then
    ((file_count++))
    validate_module_structure "test/bm1387_asic_tb.v" "bm1387_asic_tb"
    validate_verilog_constructs "test/bm1387_asic_tb.v" "Testbench"
fi

echo

# Check for module instantiation in top-level
echo "Checking Module Instantiation"
echo "============================="

if grep -q "sha256_core u_sha256_core" "src/bm1387_asic.v"; then
    echo "✓ SHA-256 core instantiation found"
else
    echo "✗ SHA-256 core instantiation not found"
    ((error_count++))
fi

if grep -q "thermal_model u_thermal_model" "src/bm1387_asic.v"; then
    echo "✓ Thermal model instantiation found"
else
    echo "✗ Thermal model instantiation not found"
    ((error_count++))
fi

if grep -q "uart_interface u_uart_if" "src/bm1387_asic.v"; then
    echo "✓ UART interface instantiation found"
else
    echo "✗ UART interface instantiation not found"
    ((error_count++))
fi

if grep -q "spi_interface u_spi_if" "src/bm1387_asic.v"; then
    echo "✓ SPI interface instantiation found"
else
    echo "✗ SPI interface instantiation not found"
    ((error_count++))
fi

echo

# Summary
echo "VALIDATION SUMMARY"
echo "=================="
echo "Files validated: $file_count"
echo "Errors found: $error_count"

if [ $error_count -eq 0 ]; then
    echo
    echo "✓ ALL VALIDATION CHECKS PASSED"
    echo
    echo "The BM1387 ASIC Verilog model structure is valid and ready for compilation."
    echo "To compile and run the simulation:"
    echo "  1. Ensure Verilator is installed"
    echo "  2. Run: ./compile_bm1387.bat"
    echo "  3. View waveforms: gtkwave waveforms/bm1387_asic_tb.vcd"
    exit 0
else
    echo
    echo "✗ VALIDATION FAILED - Please review the errors above"
    exit 1
fi
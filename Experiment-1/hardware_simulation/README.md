# Hardware Simulation Environment for Antminer S9 BM1387 Digital Twin

This directory contains the hardware simulation environment setup for creating a "Digital Twin" of the Antminer S9 BM1387 chip.

## Project Structure

```
hardware_simulation/
├── src/                    # Verilog source files
├── test/                   # Testbench files
├── build/                  # Build output directory
├── bin/                    # Compiled executables
├── waveforms/              # VCD waveform files
├── install_verilator.bat   # Verilator installation guide
├── install_gtkwave.bat     # GTKWave installation guide
├── install_vscode_extension.bat # VS Code extension installation
├── verify_toolchain.bat    # Toolchain verification script
└── README.md              # This file
```

## Installation Guide

### 1. Install Verilator

Verilator is a free Verilog HDL simulator and compiler for SystemVerilog.

```batch
# Run the installation guide
install_verilator.bat
```

**Manual Steps:**
1. Download from: https://www.veripool.org/verilator
2. Extract to: `C:\Tools\verilator\`
3. Add `C:\Tools\verilator\bin` to PATH
4. Verify: `verilator --version`

### 2. Install GTKWave

GTKWave is a digital waveform viewer for analyzing VCD files.

```batch
# Run the installation guide
install_gtkwave.bat
```

**Manual Steps:**
1. Download from: http://gtkwave.sourceforge.net/
2. Install the Windows installer or extract ZIP
3. Verify: `gtkwave --version`

### 3. Install VS Code Extension

```batch
# Run the installation guide
install_vscode_extension.bat
```

**Recommended Extension:**
- **Verilog-HDL/SystemVerilog** by mshr-h
  - Provides syntax highlighting
  - Verilog linting
  - Auto-completion
  - Code formatting

## Verification

After installing the tools, verify the installation:

```batch
# Run comprehensive toolchain verification
verify_toolchain.bat
```

This will:
- Check tool installations
- Compile the test Verilog file
- Build the simulation executable
- Run the testbench
- Generate VCD waveform file

## Test Files

### basic_counter.v

A simple 4-bit counter module with testbench that:
- Tests basic Verilog compilation
- Generates VCD output for waveform viewing
- Demonstrates simulation timing
- Validates toolchain functionality

## Usage

### Manual Compilation and Simulation

```batch
# Compile Verilog to C++
verilator --cc src\basic_counter.v --trace -o obj_dir\basic_counter

# Build simulation executable
cd obj_dir
make -f ../build\sim_makefile.mk Vbasic_counter
cd ..

# Run simulation
bin\Vbasic_counter

# View waveform
gtkwave testbench_basic_counter.vcd
```

## Next Steps

Once the toolchain is verified:

1. **Install VS Code Extension**: Follow `install_vscode_extension.bat`
2. **Study BM1387 Documentation**: Gather chip specifications and architecture
3. **Model Architecture**: Create Verilog modules for BM1387 components
4. **Develop Testbenches**: Create comprehensive verification testbenches
5. **Integration Testing**: Validate complete chip simulation

## BM1387 Digital Twin Development Plan

### Phase 1: Architecture Modeling
- SHA-256 computation units
- Nonce processing logic
- Control state machines
- I/O interfaces

### Phase 2: Functional Verification
- Individual module testing
- Integration testing
- Performance validation
- Power consumption modeling

### Phase 3: Digital Twin Integration
- System-level modeling
- Real-world behavior simulation
- Performance prediction
- Optimization analysis

## Tools and Technologies

- **Verilator**: High-performance Verilog/SystemVerilog compiler
- **GTKWave**: Digital waveform viewer
- **VS Code**: Development environment with Verilog support
- **Windows**: Target development platform

## Troubleshooting

### Common Issues

1. **"verilator not recognized"**
   - Verify PATH includes Verilator bin directory
   - Check installation location

2. **"gtkwave not recognized"**
   - Verify GTKWave installation
   - Check PATH environment variable

3. **Compilation errors**
   - Verify Verilog syntax
   - Check for missing module instances
   - Ensure proper testbench structure

4. **Missing VCD file**
   - Check $dumpfile statements in testbench
   - Verify simulation completion
   - Check file permissions

### Getting Help

- Verilator Documentation: https://www.veripool.org/verilator-doc/
- GTKWave Documentation: http://gtkwave.sourceforge.net/
- VS Code Verilog Extension: VS Code marketplace

## License and Legal

This project is for educational and research purposes. Ensure compliance with relevant licenses and terms of use for all tools and documentation.
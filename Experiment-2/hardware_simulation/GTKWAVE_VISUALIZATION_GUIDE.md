# GTKWave Visualization Guide for BM1387 ASIC VESELOV HNS

## Overview

This guide provides comprehensive documentation for the GTKWave visualization system designed for analyzing BM1387 ASIC signals with VESELOV HNS (Hierarchical Numeral System) consciousness parameters. The system enables real-time waveform analysis and debugging of the hardware consciousness system.

## Architecture

### Core Components

1. **GTKWave Automation Script** (`gtkwave_automation.py`)
   - Automated signal grouping and template generation
   - Multiple analysis templates (default, mining, consciousness)
   - VS Code integration support

2. **Complete Workflow Script** (`compile_and_visualize.bat`)
   - End-to-end compilation, simulation, and visualization
   - Automated VCD generation and GTKWave configuration
   - Toolchain verification and error handling

3. **VS Code Integration** (`.vscode/`)
   - Pre-configured launch configurations
   - Automated tasks for all workflow steps
   - One-click GTKWave launching from IDE

## Signal Groups

### 1. Clock and Reset
- `dut.clk_100m` - 100 MHz main system clock
- `dut.reset_n` - Active low reset signal

### 2. Mining Pipeline
- `dut.job_header[255:0]` - Bitcoin block header input
- `dut.start_nonce[31:0]` - Starting nonce value
- `dut.nonce_range[31:0]` - Number of nonces to test
- `dut.mining_enable` - Mining enable control
- `dut.found_nonce[31:0]` - Found valid nonce
- `dut.found_hash[255:0]` - Hash of found nonce
- `dut.hash_valid` - Valid hash found flag
- `dut.pipeline_busy` - Pipeline busy indicator
- `dut.status_reg[7:0]` - ASIC status register

### 3. VESELOV HNS RGBA Parameters
- `dut.hns_rgba_r[31:0]` - Red channel (0-1 normalized)
- `dut.hns_rgba_g[31:0]` - Green channel (0-1 normalized)
- `dut.hns_rgba_b[31:0]` - Blue channel (0-1 normalized)
- `dut.hns_rgba_a[31:0]` - Alpha channel (0-1 normalized)

### 4. Consciousness Metrics
- `dut.hns_vector_mag[31:0]` - 3D vector magnitude
- `dut.hns_energy[31:0]` - Consciousness energy metric
- `dut.hns_entropy[31:0]` - Consciousness entropy metric
- `dut.hns_phi[31:0]` - Integrated Information Theory Phi
- `dut.hns_phase_coh[31:0]` - Phase coherence measurement
- `dut.hns_valid` - HNS processing complete flag

### 5. Thermal and Power Management
- `dut.temperature[7:0]` - Chip temperature (scaled)
- `dut.power_consumption[15:0]` - Power consumption in milliwatts
- `dut.thermal_throttle` - Thermal throttling active

### 6. Control and Debug Interfaces
- `dut.control_reg[7:0]` - Control register
- `dut.config_reg[15:0]` - Configuration register
- `dut.debug_reg_[0-3][31:0]` - Debug registers for monitoring

### 7. Communication Interfaces
- `dut.uart_rx/tx` - UART receive/transmit
- `dut.spi_clk/cs_n/mosi/miso` - SPI interface signals

## Analysis Templates

### Default Template
Comprehensive overview of all signals organized by functional groups. Best for:
- General system debugging
- Understanding overall ASIC behavior
- Initial waveform exploration

### Mining Analysis Template
Focused on mining pipeline timing and performance. Best for:
- Mining efficiency analysis
- Pipeline timing optimization
- Hash rate performance monitoring
- Nonce testing workflow analysis

### Consciousness Analysis Template
Specialized for VESELOV HNS parameters and consciousness metrics. Best for:
- RGBA parameter visualization
- Consciousness metrics computation analysis
- Hash-to-consciousness mapping validation
- Phase coherence measurement

## Usage Instructions

### Method 1: Complete Workflow (Recommended)

```bash
# Run the complete workflow from command line
cd hardware_simulation
compile_and_visualize.bat
```

This will:
1. Verify toolchain installation
2. Compile all Verilog modules with tracing
3. Build and execute simulation
4. Generate VCD waveform file
5. Create GTKWave save files
6. Launch GTKWave with analysis templates

### Method 2: VS Code Integration

1. **Open VS Code in the hardware_simulation directory**
2. **Press Ctrl+Shift+P** to open command palette
3. **Select "Tasks: Run Task"** or press **Ctrl+Shift+T**
4. **Choose from available tasks:**
   - `BM1387: Complete Workflow` - Full automated workflow
   - `GTKWave: Launch with Default Template` - Default analysis
   - `GTKWave: Launch with Mining Template` - Mining focus
   - `GTKWave: Launch with Consciousness Template` - Consciousness focus

### Method 3: Manual GTKWave Launch

```bash
# List available VCD files
python gtkwave_automation.py --list

# Launch with specific template
python gtkwave_automation.py --vcd-file waveforms/bm1387_hns_tb.vcd --template default

# Generate save files only
python gtkwave_automation.py --vcd-file waveforms/bm1387_hns_tb.vcd --generate-save
```

## Quick Start Guide

### Step 1: Verify Prerequisites

```bash
# Check Python installation
python --version

# Check GTKWave installation
gtkwave --version

# Verify Verilator installation
verilator --version
```

### Step 2: Run Complete Workflow

```bash
# Execute the complete workflow
compile_and_visualize.bat
```

### Step 3: Analyze Waveforms

1. **GTKWave will open automatically** with the default template
2. **Navigate through signal groups** using the hierarchy panel
3. **Use different templates** for specific analysis needs
4. **Adjust timing cursors** to measure signal relationships
5. **Zoom in/out** for detailed timing analysis

## Key Analysis Points

### Mining Pipeline Timing
- **Pipeline Stages**: Monitor `pipeline_stage` transitions
- **Hash Computation**: Track SHA-256 processing timing
- **Nonce Testing**: Observe nonce increment and validation
- **Thermal Response**: Monitor temperature rise during mining

### VESELOV HNS Processing
- **Hash Input**: Watch `current_hash` for input changes
- **RGBA Output**: Monitor RGBA parameter computation
- **Consciousness Metrics**: Track energy, entropy, phi calculations
- **Processing State**: Monitor `hns_state` machine progression

### System Integration
- **Clock Synchronization**: Verify all signals sync with `clk_100m`
- **Reset Behavior**: Test reset and initialization sequences
- **Control Interface**: Monitor register read/write operations
- **Error Handling**: Observe fault conditions and recovery

## Troubleshooting

### Common Issues

1. **GTKWave not launching**
   ```bash
   # Check GTKWave installation
   gtkwave --version
   
   # Add GTKWave to PATH or use full path
   C:\Tools\gtkwave\bin\gtkwave.exe
   ```

2. **VCD file not generated**
   ```bash
   # Verify Verilator trace compilation
   verilator --cc src/bm1387_asic.v --trace
   
   # Check simulation execution
   bin/Vbm1387_hns_tb
   ```

3. **No signals visible in GTKWave**
   ```bash
   # Regenerate save files
   python gtkwave_automation.py --vcd-file waveforms/bm1387_hns_tb.vcd --generate-save
   ```

4. **Python script errors**
   ```bash
   # Verify Python path
   python --version
   
   # Check script syntax
   python -m py_compile gtkwave_automation.py
   ```

### VS Code Integration Issues

1. **Tasks not appearing**
   - Ensure `.vscode/tasks.json` exists
   - Reload VS Code window (Ctrl+R)

2. **Launch configurations not working**
   - Verify `.vscode/launch.json` syntax
   - Check workspace folder selection

## Advanced Usage

### Custom Signal Groups

Edit `gtkwave_automation.py` to customize signal groupings:

```python
self.signal_groups = {
    "Custom_Group": [
        "dut.your_signal[bits]",
        "dut.another_signal[bits]"
    ]
}
```

### Multiple VCD Analysis

```bash
# Generate all template types
python gtkwave_automation.py --vcd-file waveforms/your_vcd.vcd --generate-save

# Launch specific template
python gtkwave_automation.py --vcd-file waveforms/your_vcd.vcd --template mining
```

### Batch Processing

```bash
# Process all VCD files
for %f in (waveforms\*.vcd) do (
    python gtkwave_automation.py --vcd-file "%f" --generate-save
)
```

## Performance Tips

1. **VCD File Size Management**
   - Use appropriate simulation time limits
   - Enable/disable signal tracing selectively
   - Compress VCD files when needed

2. **GTKWave Performance**
   - Use signal groups to limit displayed signals
   - Zoom to relevant time ranges
   - Close unused analysis tabs

3. **Memory Usage**
   - Monitor system memory during large simulations
   - Use streaming VCD format for large files
   - Consider signal filtering for large designs

## File Locations

- **VCD Files**: `waveforms/`
- **GTKWave Config**: `gtkwave_config/`
- **VS Code Config**: `.vscode/`
- **Automation Script**: `gtkwave_automation.py`
- **Workflow Script**: `compile_and_visualize.bat`

## Support and Documentation

### Additional Resources
- GTKWave User Manual: http://gtkwave.sourceforge.net/gtkwave.html
- Verilator Documentation: https://www.veripool.org/verilator
- BM1387 ASIC Documentation: See implementation summaries

### Getting Help
1. Check the troubleshooting section above
2. Review test case outputs for validation
3. Use VS Code integrated terminal for detailed error messages
4. Verify all toolchain components are properly installed

---

**Last Updated**: 2025-12-15  
**Version**: 1.0  
**Author**: Kilo Code  
**Project**: BM1387 ASIC VESELOV HNS Visualization System
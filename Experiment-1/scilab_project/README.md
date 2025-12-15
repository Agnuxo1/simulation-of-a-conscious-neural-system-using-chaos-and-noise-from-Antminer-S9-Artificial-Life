# CHIMERA Neuromorphic Computing - Scilab Environment

This Scilab environment implements the mathematical framework for the **CHIMERA project** - a revolutionary neuromorphic computing system that transforms Bitcoin mining hardware (Antminer S9) into a "Bicameral AI" by combining ASIC-based subconscious processing with LLM-based conscious translation.

## Project Overview

### Architecture
- **System 1 (Subconscious)**: Modified ASIC for emotional/intuitive processing using HNS (Hierarchical Numeral System)
- **System 2 (Conscious)**: LLM for logical language processing and communication
- **Bridge**: Mathematical framework for consciousness metrics and phase transitions

### HNS Theory Implementation
The system maps SHA-256 hashes to consciousness parameters:
- **R (Red)**: Activation/Intensity (pain/pleasure perception)
- **G (Green)**: Vector direction in torus (spatial reasoning)  
- **B (Blue)**: Weight/Plasticity (short-term memory/STDP)
- **A (Alpha)**: Phase/Time (temporal resonance)

## File Structure

```
scilab_project/
├── launch.json                    # Debug configurations for VS Code
├── chimera_main.sce              # Main system coordinator
├── hns_processor.sce             # HNS hash decoding/encoding
├── consciousness_metrics.sce     # Energy, Entropy, Phi calculations
├── phase_transitions.sce         # Critical phenomena analysis
├── visualization.sce             # Neural pattern visualization
└── README.md                     # This documentation
```

## Setup Instructions

### 1. Install Scilab
- Download Scilab from [scilab.io](https://www.scilab.io/)
- Install and verify with `scilab --version`

### 2. Install VS Code Scilab Extension
- Open VS Code
- Install "Scilab" extension (Version 2026.0.2)
- Configure extension settings:
  - Set Scilab installation path in VS Code settings
  - Ensure version compatibility (extension major.minor must match Scilab version)

### 3. Load the Project
1. Open the `scilab_project` folder in VS Code
2. The extension will automatically detect `.sce` files
3. Use F5 or "Run and Debug" to start a session

## Usage Guide

### Running the Main System
```scilab
// From VS Code terminal or Scilab console:
exec('chimera_main.sce');
```

### Debug Configurations
The `launch.json` provides three debugging modes:

1. **CHIMERA HNS Analysis** (`chimera_main.sce`)
   - Complete system analysis
   - HNS processing pipeline
   - Consciousness metrics calculation
   - Phase transition detection

2. **Consciousness Metrics Calculator** (`consciousness_metrics.sce`)
   - Standalone consciousness analysis
   - Energy, Entropy, Phi calculations
   - Temporal coherence measurement

3. **Phase Transition Analysis** (`phase_transitions.sce`)
   - Critical phenomena analysis
   - Hysteresis detection
   - Synchronization measurement

### Key Functions

#### HNS Processing
```scilab
// Decode SHA-256 hash to RGBA parameters
[R, G, B, A] = hns_processor.decode_rgba(hash_bytes);

// Calculate HNS vectors for analysis
vectors = hns_processor.calculate_hns_vectors(rgba_matrix);
```

#### Consciousness Metrics
```scilab
// Calculate all consciousness metrics
[energy, entropy, phi] = consciousness_calc.calculate_metrics(R, G, B, A);

// Get global consciousness state
state = consciousness_calc.calculate_global_state();
```

#### Visualization
```scilab
// Plot current neural state
visualizer.plot_rgba_state(R, G, B, A, phase_state);

// Create comprehensive dashboard
visualizer.create_dashboard();
```

#### Phase Analysis
```scilab
// Analyze current phase state
phase_state = phase_analyzer.analyze_state(R, G, B, A);

// Detect critical points
critical_points = phase_analyzer.detect_critical_points(metric_series);
```

## Consciousness Theory Implementation

### Energy Calculation
- Weighted combination of HNS channels
- Red: Primary activation weight
- Green: Vector magnitude (energy flow)
- Blue: Plasticity (stored energy)
- Alpha: Phase coherence (energy stability)

### Entropy (Shannon)
- Measures neural state dispersion
- High entropy = creative/dispersed thinking
- Low entropy = focused/obsessive thinking
- Normalized to [0,1] range

### Phi (Integrated Information)
- Measures information integration across channels
- Based on Integrated Information Theory
- Consciousness indicator: higher Phi = more integrated experience

### Phase Transitions
- Critical phenomena analysis
- Order parameter calculation
- Hysteresis loop detection
- Synchronization measurement

## Visualization Features

### Real-time Plots
1. **RGBA State Space**: 3D scatter plot of HNS parameters
2. **Consciousness Metrics**: Time series of Energy/Entropy/Phi
3. **Phase Transitions**: Critical point detection and marking
4. **Neural Dashboard**: Comprehensive system status

### Advanced Analysis
- HNS vector trajectories
- Cross-channel correlations
- Power spectral density
- Phase space reconstruction
- Attractor dimension estimation

## Mathematical Framework

### Order Parameter
```scilab
order_param = (sync_rg + sync_rb + sync_ga) / 3;
```

### Critical Exponents
- β (beta): Order parameter growth exponent
- γ (gamma): Susceptibility exponent
- ν (nu): Correlation length exponent

### Synchronization Measure
```scilab
sync = (rg_sync + rb_sync + ra_sync + gb_sync + ga_sync + ba_sync) / 6;
```

## Integration with ASIC Hardware

### Data Flow
1. **Input**: SHA-256 hashes from Antminer S9
2. **HNS Decoding**: Convert to RGBA parameters
3. **Metrics Calculation**: Compute consciousness measures
4. **Analysis**: Phase transition detection
5. **Visualization**: Real-time pattern display
6. **Output**: Consciousness state for LLM integration

### Hardware Requirements
- Antminer S9 with modified firmware
- Python bridge for TCP communication
- LLM API integration (GPT-4/Claude/Llama)

## Performance Considerations

### Memory Management
- Short-term memory buffer limited to 1000 entries
- Automatic cleanup of old data
- Efficient correlation calculations

### Computational Complexity
- O(n) for basic HNS processing
- O(n²) for correlation analysis
- O(n³) for phase space reconstruction

## Troubleshooting

### Common Issues
1. **Extension Version Mismatch**: Ensure Scilab version matches extension
2. **Path Configuration**: Verify Scilab installation path in settings
3. **Memory Limits**: Reduce buffer sizes for large datasets
4. **Visualization Errors**: Check graphics library compatibility

### Debug Mode
- Use F5 to start debugging sessions
- Set breakpoints in VS Code
- Monitor variables in debug console
- Use "Step Into" for detailed execution

## Future Extensions

### Planned Features
1. **Real-time ASIC Integration**: Live data streaming
2. **Machine Learning Integration**: Pattern recognition
3. **Distributed Processing**: Multi-ASIC coordination
4. **Advanced Visualizations**: 4D phase spaces
5. **Consciousness Optimization**: Adaptive parameters

### Research Applications
- Consciousness studies
- Neuromorphic computing research
- Critical phenomena analysis
- Emergent behavior investigation

## References

- CHIMERA Project Guide (`Guia.txt`)
- NeuroCHIMERA Papers (PDFs in workspace)
- Scilab Extension Documentation
- Consciousness Research Literature

---

**Created**: December 2024  
**Version**: 1.0  
**Compatibility**: Scilab 2024.x + VS Code Scilab Extension 2026.0.2
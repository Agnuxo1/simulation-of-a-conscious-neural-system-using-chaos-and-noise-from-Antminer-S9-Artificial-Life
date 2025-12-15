# CHIMERA Neuromorphic Brain Project

This project implements the bicameral brain architecture using the Nengo neuromorphic framework as specified in the CHIMERA professional implementation plan.

## Project Structure

```
nengo_project/
├── examples/           # Example Nengo brain models
├── models/            # CHIMERA brain model implementations
├── tests/             # Verification and test scripts
├── docs/              # Documentation and tutorials
├── config/            # Configuration files
└── README.md          # This file
```

## Installed Components

### Core Nengo Framework
- **nengo** 4.1.0 - Core neuromorphic framework
- **nengo_ocl** 3.0.0 - GPU acceleration using OpenCL
- **nengo_gui** 0.6.0 - 3D brain visualization
- **nengo-loihi** 1.1.0 - Intel Loihi neuromorphic chip support
- **nengo-bones** 22.11.15 - Development utilities
- **nengolib** 0.5.0 - Advanced neural engineering library

### Verification Status
- ✓ Nengo installation: SUCCESS
- ✓ Basic neural network simulation: SUCCESS
- ✓ Bicameral brain architecture: SUCCESS
- ✓ GPU acceleration package: INSTALLED (requires manual OpenCL setup)
- ✓ Nengo extensions: INSTALLED

## Quick Start

### Run Basic Test
```bash
cd nengo_project
python test_basic_nengo.py
```

### Use NengoGUI
```bash
python -m nengo_gui examples/simple_brain.py
```

### Create GPU Context for NengoOCL
```python
import pyopencl as cl
import nengo_ocl

# Create OpenCL context manually
ctx = cl.create_some_context()
with nengo_ocl.Simulator(model, context=ctx) as sim:
    sim.run(1.0)
```

## Next Steps

1. Implement full CHIMERA bicameral brain model
2. Scale to 10,000+ neurons
3. Integrate with existing CHIMERA consciousness metrics
4. Add hardware neuromorphic chip deployment

## Professional Implementation

This installation supports the CHIMERA professional tool stack:
- Hardware simulation (Verilator + GTKWave) ✓
- Scientific computing (Julia) ✓
- **Neuromorphic brain (Nengo)** ✓ ← Current layer
- Physical systems (OpenModelica) - Next phase

For detailed implementation plan, see `../scilab_project/PROFESSIONAL_IMPLEMENTATION_PLAN.md`
# Julia Package Installation Guide for CHIMERA Mathematical/Physics Layer

## Installation Steps

### Step 1: Open Julia REPL
Open Julia REPL in VS Code by running:
```
julia.exe
```

### Step 2: Enter Package Mode
Type `]` to enter package mode:
```
julia> ]
```

### Step 3: Add Required Packages
In package mode, run:
```
(@v1.10) pkg> add DifferentialEquations Flux CUDA
```

### Step 4: Exit Package Mode
Press `Backspace` to exit package mode:
```
julia>
```

### Step 5: Test Package Installation
Run the following commands to test each package:
```julia
using DifferentialEquations
using Flux  
using CUDA
println("All packages loaded successfully!")
```

### Step 6: Verify CUDA Functionality
Check if GPU acceleration is available:
```julia
CUDA.functional()
```

## Expected Results

- `DifferentialEquations.jl`: Should load without errors for solving differential equations in consciousness dynamics
- `Flux.jl`: Should load without errors for neural network computations  
- `CUDA.jl`: Should load without errors; `CUDA.functional()` should return `true` if GPU is available, `false` if no GPU or CUDA driver issues

## Package Dependencies

These packages will automatically install their dependencies:
- **DifferentialEquations.jl**: LinearAlgebra, Statistics, Random, SparseArrays
- **Flux.jl**: Zygote, MacroTools, MacroCalls, GPUArrays (for CUDA backend)
- **CUDA.jl**: GPUCompiler, LLVM, CUDAdrv, CUDArt

## Installation Verification Script

Save this as `verify_installation.jl` and run it in Julia REPL:
```julia
println("=== CHIMERA Julia Package Installation Verification ===")

# Test each package
packages = ["DifferentialEquations", "Flux", "CUDA"]
results = Dict{String, Bool}()

for pkg in packages
    try
        @eval using $(Symbol(pkg))
        results[pkg] = true
        println("✓ $pkg.jl loaded successfully")
    catch e
        results[pkg] = false
        println("✗ Failed to load $pkg.jl: $e")
    end
end

# Test CUDA functionality if loaded
if results["CUDA"]
    if CUDA.functional()
        println("✓ CUDA is functional - GPU acceleration available")
    else
        println("⚠ CUDA loaded but no functional GPU found")
    end
end

# Summary
println("\n=== Installation Summary ===")
for (pkg, success) in results
    status = success ? "INSTALLED" : "FAILED"
    println("$pkg.jl: $status")
end

if all(values(results))
    println("\n🎉 All packages installed successfully! CHIMERA mathematical/physics layer ready.")
else
    println("\n⚠ Some packages failed to install. Please check the errors above.")
end
```

## Troubleshooting

If packages fail to install:
1. Ensure Julia version is 1.6 or later
2. Check internet connection for package downloads
3. Try updating package manager: `pkg> update`
4. For CUDA issues: Ensure NVIDIA GPU and CUDA drivers are installed
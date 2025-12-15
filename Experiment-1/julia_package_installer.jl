# Julia Package Installation Script for CHIMERA Mathematical/Physics Layer
# This script installs and tests the required packages

println("Starting Julia package installation for CHIMERA...")

# Add the required packages
println("Adding packages: DifferentialEquations, Flux, CUDA")
using Pkg
Pkg.add(["DifferentialEquations", "Flux", "CUDA"])

println("\nTesting package imports...")

# Test DifferentialEquations
println("Testing DifferentialEquations...")
try
    using DifferentialEquations
    println("✓ DifferentialEquations.jl loaded successfully")
catch e
    println("✗ Failed to load DifferentialEquations.jl: $e")
end

# Test Flux
println("Testing Flux...")
try
    using Flux
    println("✓ Flux.jl loaded successfully")
catch e
    println("✗ Failed to load Flux.jl: $e")
end

# Test CUDA
println("Testing CUDA...")
try
    using CUDA
    println("✓ CUDA.jl loaded successfully")
    if CUDA.functional()
        println("✓ CUDA is functional - GPU acceleration available")
    else
        println("⚠ CUDA loaded but no functional GPU found")
    end
catch e
    println("✗ Failed to load CUDA.jl: $e")
end

println("\nJulia package installation and testing completed!")
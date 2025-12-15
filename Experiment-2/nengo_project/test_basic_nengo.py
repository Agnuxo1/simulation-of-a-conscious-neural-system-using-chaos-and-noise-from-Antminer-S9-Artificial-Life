#!/usr/bin/env python3
"""
Basic Nengo Neural Network Example
==================================

This script verifies the Nengo installation by creating a simple neural network
that demonstrates basic functionality for the CHIMERA bicameral brain architecture.
"""

import nengo
import numpy as np
import nengo_ocl
from nengo.neurons import LIF
import time

def test_basic_nengo():
    """Test basic Nengo functionality"""
    print("=== Basic Nengo Installation Test ===")
    
    # Create a simple model
    model = nengo.Network(label="Basic Test Network")
    
    with model:
        # Create input node (sinusoidal input)
        input_node = nengo.Node(
            output=lambda t: np.sin(2 * np.pi * t),
            label="Sinusoidal Input"
        )
        
        # Create neural ensemble
        ensemble = nengo.Ensemble(
            n_neurons=100,
            dimensions=1,
            neuron_type=LIF(),
            label="Test Ensemble"
        )
        
        # Create connections
        nengo.Connection(input_node, ensemble)
        
        # Add probes to collect data
        input_probe = nengo.Probe(input_node)
        ensemble_probe = nengo.Probe(ensemble, synapse=0.01)
    
    # Test CPU simulation
    print("Testing CPU simulation...")
    start_time = time.time()
    with nengo.Simulator(model) as sim:
        sim.run(1.0)  # Run for 1 second
    cpu_time = time.time() - start_time
    print(f"CPU simulation completed in {cpu_time:.3f} seconds")
    
    # Test GPU acceleration if available
    print("Testing NengoOCL GPU acceleration...")
    try:
        start_time = time.time()
        with nengo_ocl.Simulator(model) as sim:
            sim.run(1.0)  # Run for 1 second
        gpu_time = time.time() - start_time
        print(f"GPU simulation completed in {gpu_time:.3f} seconds")
        print(f"GPU speedup: {cpu_time/gpu_time:.2f}x")
        gpu_available = True
    except Exception as e:
        print(f"GPU acceleration not available: {e}")
        gpu_available = False
    
    return {
        'cpu_time': cpu_time,
        'gpu_time': gpu_time if gpu_available else None,
        'gpu_available': gpu_available,
        'success': True
    }

def test_consciousness_architecture():
    """Test basic consciousness architecture components"""
    print("\n=== Consciousness Architecture Test ===")
    
    # Create a simplified bicameral brain model
    model = nengo.Network(label="Simplified CHIMERA Brain")
    
    with model:
        # System 1: Subcortical processing
        subcortical_input = nengo.Node(
            output=lambda t: [np.sin(t), np.cos(t), np.random.random(), np.random.random()],
            label="ASIC Input"
        )
        
        subcortical = nengo.Ensemble(
            n_neurons=200,
            dimensions=4,  # RGBA dimensions
            neuron_type=LIF(),
            label="Subcortical Processing"
        )
        
        # System 2: Cortical processing  
        cortical_input = nengo.Node(
            output=lambda t: np.random.random(4),
            label="Conscious Input"
        )
        
        cortical = nengo.Ensemble(
            n_neurons=150,
            dimensions=4,
            neuron_type=LIF(),
            label="Cortical Processing"
        )
        
        # Bridge connection (bottom-up)
        nengo.Connection(subcortical, cortical, 
                        transform=np.eye(4) * 0.5,
                        label="Bottom-up Bridge")
        
        # Connection from inputs
        nengo.Connection(subcortical_input, subcortical)
        nengo.Connection(cortical_input, cortical)
        
        # Add probes
        subcortical_probe = nengo.Probe(subcortical, synapse=0.01)
        cortical_probe = nengo.Probe(cortical, synapse=0.01)
    
    # Run simulation
    print("Testing bicameral architecture...")
    start_time = time.time()
    with nengo.Simulator(model) as sim:
        sim.run(2.0)  # Run for 2 seconds
    sim_time = time.time() - start_time
    print(f"Bicameral simulation completed in {sim_time:.3f} seconds")
    
    return {
        'simulation_time': sim_time,
        'architecture_type': 'bicameral',
        'success': True
    }

def main():
    """Run all tests"""
    print("Nengo Neuromorphic Framework Installation Verification")
    print("=" * 60)
    
    # Test basic functionality
    basic_results = test_basic_nengo()
    
    # Test consciousness architecture
    consciousness_results = test_consciousness_architecture()
    
    # Summary
    print("\n=== Installation Summary ===")
    print(f"Nengo installation: {'✓ SUCCESS' if basic_results['success'] else '✗ FAILED'}")
    print(f"GPU acceleration: {'✓ AVAILABLE' if basic_results['gpu_available'] else '✗ NOT AVAILABLE'}")
    print(f"Consciousness architecture: {'✓ SUCCESS' if consciousness_results['success'] else '✗ FAILED'}")
    
    if basic_results['gpu_available']:
        print(f"GPU speedup: {basic_results['cpu_time']/basic_results['gpu_time']:.2f}x")
    
    print("\nNengo installation verification complete!")
    return basic_results, consciousness_results

if __name__ == "__main__":
    main()
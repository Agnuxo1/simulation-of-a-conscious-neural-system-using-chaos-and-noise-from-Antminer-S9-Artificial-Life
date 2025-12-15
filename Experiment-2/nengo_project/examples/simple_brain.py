#!/usr/bin/env python3
"""
Simple Brain Example for CHIMERA
================================

This example demonstrates basic Nengo usage for the CHIMERA bicameral brain architecture.
Run this with: python -m nengo_gui simple_brain.py
"""

import nengo
import numpy as np
from nengo.neurons import LIF

# Create the CHIMERA brain model
model = nengo.Network(label="CHIMERA Simple Brain")

with model:
    # Input: Simulated ASIC hash processing
    asic_input = nengo.Node(
        output=lambda t: [np.sin(t*2), np.cos(t*2), np.random.random(), np.random.random()],
        label="ASIC Input"
    )
    
    # Subcortical system (System 1): Fast, intuitive processing
    subcortical = nengo.Ensemble(
        n_neurons=100,
        dimensions=4,  # RGBA color space
        neuron_type=LIF(),
        label="Subcortical Processing"
    )
    
    # Cortical system (System 2): Slow, conscious processing
    cortical = nengo.Ensemble(
        n_neurons=80,
        dimensions=4,
        neuron_type=LIF(),
        label="Cortical Processing"
    )
    
    # Bottom-up connection: Subcortical influences cortical
    nengo.Connection(
        subcortical, cortical,
        transform=np.eye(4) * 0.3,
        label="Bottom-up Bridge"
    )
    
    # Input connections
    nengo.Connection(asic_input, subcortical)
    
    # Probes to observe activity
    asic_probe = nengo.Probe(asic_input, sample_every=0.1)
    subcortical_probe = nengo.Probe(subcortical, synapse=0.01, sample_every=0.1)
    cortical_probe = nengo.Probe(cortical, synapse=0.01, sample_every=0.1)

# Run simulation
if __name__ == "__main__":
    print("Running CHIMERA Simple Brain Example...")
    print("Components:")
    print("- ASIC Input: Simulated hash processing")
    print("- Subcortical: 100 neurons, 4D RGBA")
    print("- Cortical: 80 neurons, 4D RGBA")
    print("- Bottom-up bridge connection")
    
    with nengo.Simulator(model) as sim:
        sim.run(2.0)  # Run for 2 seconds
    
    print("Simulation completed successfully!")
    print(f"Time steps: {len(sim.trange())}")
    print(f"ASIC input shape: {sim.data[asic_probe].shape}")
    print(f"Subcortical activity shape: {sim.data[subcortical_probe].shape}")
    print(f"Cortical activity shape: {sim.data[cortical_probe].shape}")
    
    # Analyze the results
    print("\nActivity Analysis:")
    print(f"ASIC input range: [{np.min(sim.data[asic_probe]):.3f}, {np.max(sim.data[asic_probe]):.3f}]")
    print(f"Subcortical mean activity: {np.mean(sim.data[subcortical_probe]):.3f}")
    print(f"Cortical mean activity: {np.mean(sim.data[cortical_probe]):.3f}")
    
    print("\nTo visualize in 3D, run:")
    print("python -m nengo_gui examples/simple_brain.py")
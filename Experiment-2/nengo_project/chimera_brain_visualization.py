#!/usr/bin/env python3
"""
NengoGUI Visualization Script for CHIMERA Brain
==============================================

This script provides 3D visualization of the bicameral brain architecture
using NengoGUI. Run with: python -m nengo_gui chimera_brain_visualization.py

Author: Kilo Code
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import nengo
import nengo_gui
from chimera_brain import ChimeraBrain
import numpy as np
import matplotlib.pyplot as plt

def create_visualization_model():
    """
    Create a Nengo model specifically optimized for GUI visualization.
    This model includes visual representations of the brain components.
    """
    # Create the CHIMERA brain
    brain = ChimeraBrain(n_neurons=1000, enable_gui=True)
    
    # Create a visualization-optimized model
    viz_model = nengo.Network(label="CHIMERA Brain Visualization")
    
    with viz_model:
        # Add visualization nodes for better GUI display
        viz_input = nengo.Node(lambda t: [np.sin(t*2), np.cos(t*2), 
                                         np.sin(t*3), np.cos(t*3)], 
                              label="Visualization Input")
        
        # Create a simplified version for visualization
        brain_visual = nengo.Ensemble(
            n_neurons=200,
            dimensions=4,
            neuron_type=nengo.neurons.LIF(),
            label="Brain Visualization"
        )
        
        # Visual processing pipeline
        nengo.Connection(viz_input, brain_visual)
        
        # Add visual feedback
        viz_output = nengo.Node(lambda t, x: x, 
                               size_in=4, 
                               label="Visualization Output")
        
        nengo.Connection(brain_visual, viz_output)
        
        # Add probes for monitoring
        viz_input_probe = nengo.Probe(viz_input, sample_every=0.1)
        brain_viz_probe = nengo.Probe(brain_visual, synapse=0.01, sample_every=0.1)
        viz_output_probe = nengo.Probe(viz_output, sample_every=0.1)
        
        # Store probes for analysis
        viz_model.viz_input_probe = viz_input_probe
        viz_model.brain_viz_probe = brain_viz_probe
        viz_model.viz_output_probe = viz_output_probe
    
    return viz_model, brain

def analyze_brain_activity():
    """Analyze and visualize brain activity patterns."""
    print("Analyzing CHIMERA brain activity patterns...")
    
    # Create and run brain simulation
    brain = ChimeraBrain(n_neurons=2000, enable_gui=True)
    results = brain.simulate(duration=5.0)
    
    # Create activity analysis plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('CHIMERA Brain Activity Analysis', fontsize=16)
    
    time = results['time']
    
    # ASIC Input Activity
    axes[0, 0].plot(time, results['asic_input'])
    axes[0, 0].set_title('ASIC Input (VESELOV HNS RGBA)')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Activity')
    axes[0, 0].legend(['R', 'G', 'B', 'A'])
    
    # Subcortical Activity
    axes[0, 1].plot(time, np.mean(results['subcortical'], axis=1))
    axes[0, 1].set_title('Subcortical Processing')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Mean Activity')
    
    # Emotional States
    axes[0, 2].plot(time, results['emotional'])
    axes[0, 2].set_title('Emotional States')
    axes[0, 2].set_xlabel('Time (s)')
    axes[0, 2].set_ylabel('Activity')
    axes[0, 2].legend(['Energy', 'Valence', 'Arousal'])
    
    # Working Memory Activity
    axes[1, 0].plot(time, np.mean(results['working_memory'], axis=1))
    axes[1, 0].set_title('Working Memory')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Mean Activity')
    
    # Executive Control
    axes[1, 1].plot(time, results['executive'])
    axes[1, 1].set_title('Executive Control')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Activity')
    axes[1, 1].legend(['Attention', 'Focus', 'Inhibition', 'Integration'])
    
    # Consciousness Metrics
    axes[1, 2].plot(time, results['consciousness'])
    axes[1, 2].set_title('Consciousness Metrics')
    axes[1, 2].set_xlabel('Time (s)')
    axes[1, 2].set_ylabel('Metric Value')
    axes[1, 2].legend(['Energy', 'Entropy', 'Phi', 'Phase State'])
    
    plt.tight_layout()
    plt.savefig('nengo_project/brain_activity_analysis.png', dpi=300, bbox_inches='tight')
    print("Brain activity analysis saved to brain_activity_analysis.png")
    
    return results

def create_gui_config():
    """Create NengoGUI configuration for optimal visualization."""
    gui_config = """
# NengoGUI Configuration for CHIMERA Brain
# ========================================

# Window settings
Config.topbar = True
Config.saved_panels = [
    "Components: ['brain_visual']",
    "Simulator: ['time', 'brain_activity']",
    "Plots: ['brain_activity']"
]

# Panel configurations
Panel.brain_visual = {
    'type': 'nengo_gui.page.builtins.components.SimControl',
    'pos': (0, 0),
    'size': (0.5, 0.8),
    'config': {
        'show_mouse': False,
        'show_axes': True,
        'show_grid': True,
        'dt': 0.001,
        'sim_time': 10.0
    }
}

Panel.brain_activity = {
    'type': 'nengo_gui.page.builtins.components.Probe',
    'pos': (0.5, 0),
    'size': (0.5, 0.8),
    'config': {
        'show_legend': True,
        'show_zero_line': True,
        'autoscale': True
    }
}

# Styling
Style.background_color = 'white'
Style.border_color = 'black'
Style.font_size = 12
Style.title_font_size = 14
"""
    
    with open('nengo_project/gui_config.cfg', 'w') as f:
        f.write(gui_config)
    
    print("GUI configuration saved to gui_config.cfg")

def run_gui_visualization():
    """Launch the NengoGUI with CHIMERA brain visualization."""
    print("Starting NengoGUI visualization...")
    print("The GUI will open in your browser.")
    print("Close the browser window to stop the visualization.")
    
    try:
        # Create GUI and start it
        gui = nengo_gui.GUI()
        gui.start()
    except Exception as e:
        print(f"Error starting GUI: {e}")
        print("Make sure nengo_gui is installed: pip install nengo_gui")

def main():
    """Main function to run brain visualization and analysis."""
    print("CHIMERA Brain 3D Visualization")
    print("=" * 40)
    
    # Create brain instance
    print("1. Creating CHIMERA brain instance...")
    brain = ChimeraBrain(n_neurons=1500, enable_gui=True)
    
    # Analyze brain activity
    print("2. Running brain activity analysis...")
    results = analyze_brain_activity()
    
    # Print summary metrics
    print("3. Consciousness Metrics Summary:")
    metrics = results['metrics']
    print(f"   Average Energy: {metrics['avg_energy']:.4f}")
    print(f"   Average Entropy: {metrics['avg_entropy']:.4f}")
    print(f"   Average Phi: {metrics['avg_phi']:.4f}")
    print(f"   Phase Transitions: {metrics['phase_transitions']}")
    print(f"   Simulation Stability: {metrics['simulation_stability']:.4f}")
    
    # Create GUI configuration
    print("4. Creating GUI configuration...")
    create_gui_config()
    
    # Launch visualization
    print("5. Launching 3D visualization...")
    print("   Run: python -m nengo_gui chimera_brain_visualization.py")
    
    print("\nVisualization setup complete!")
    print("\nTo visualize the brain in 3D:")
    print("1. Run: python -m nengo_gui chimera_brain_visualization.py")
    print("2. Open browser to localhost:8080")
    print("3. Explore the bicameral architecture interactively")
    
    # Optional: Launch GUI immediately
    response = input("\nLaunch GUI now? (y/n): ").lower().strip()
    if response in ['y', 'yes']:
        run_gui_visualization()

if __name__ == "__main__":
    main()
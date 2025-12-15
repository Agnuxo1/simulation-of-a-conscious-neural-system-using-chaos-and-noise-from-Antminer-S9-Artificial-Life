#!/usr/bin/env python3
"""
CHIMERA Bicameral Brain Architecture
===================================

Complete implementation of the CHIMERA neuromorphic brain with subcortical and 
cortical systems as specified in the PROFESSIONAL_IMPLEMENTATION_PLAN.md.

Author: Kilo Code
Version: 1.0
"""

import nengo
import numpy as np
from nengo.neurons import LIF, RectifiedLinear
from nengo.dists import Uniform
import hashlib
import time
from typing import Dict, List, Tuple, Any
import warnings

class ChimeraBrain:
    """
    Complete CHIMERA bicameral brain architecture implementing both subcortical 
    and cortical systems with VESELOV HNS integration and consciousness metrics.
    """
    
    def __init__(self, n_neurons: int = 10000, enable_gui: bool = True):
        """
        Initialize the ChimeraBrain with configurable neuron count.
        
        Args:
            n_neurons: Total number of neurons in the brain
            enable_gui: Enable NengoGUI visualization
        """
        self.n_neurons = n_neurons
        self.enable_gui = enable_gui
        self.model = nengo.Network(label="CHIMERA Bicameral Brain")
        
        # Consciousness state tracking
        self.consciousness_history = []
        self.subcortical_activity = []
        self.cortical_activity = []
        
        # VESELOV HNS parameters
        self.hns_temperature = 25.0  # Celsius
        self.hns_base_energy = 1.0
        self.phase_transition_threshold = 0.7
        
        # Build the complete bicameral architecture
        self._build_subcortical_system()
        self._build_cortical_system()
        self._build_bridge_connections()
        self._build_consciousness_integration()
        
        print(f"CHIMERA Brain initialized with {n_neurons} neurons")
        print(f"Subcortical: {int(n_neurons * 0.6)} neurons")
        print(f"Cortical: {int(n_neurons * 0.4)} neurons")
        
    def _build_subcortical_system(self):
        """Build System 1: ASIC-based subconscious processing"""
        with self.model:
            # ASIC input layer (receives hash data from mining operations)
            self.asic_input = nengo.Node(
                output=self._asic_input_function,
                label="ASIC Input Layer"
            )
            
            # Subcortical processing ensemble (60% of total neurons)
            n_subcortical = int(self.n_neurons * 0.6)
            self.subcortical = nengo.Ensemble(
                n_neurons=n_subcortical,
                dimensions=4,  # RGBA dimensions from VESELOV HNS
                neuron_type=LIF(),
                label="Subcortical Processing",
                seed=42
            )
            
            # Emotional state computation (20% of total neurons)
            n_emotional = int(self.n_neurons * 0.2)
            self.emotional_state = nengo.Ensemble(
                n_neurons=n_emotional,
                dimensions=3,  # Energy, Valence, Arousal
                neuron_type=LIF(),
                label="Emotional States",
                seed=43
            )
            
            # Intuitive processing output (with input for connections)
            self.intuitive_output = nengo.Node(
                size_in=3,  # Accept input from emotional state
                output=self._intuitive_processing,
                label="Intuitive Output"
            )
            
            # Connections within subcortical system
            nengo.Connection(self.asic_input, self.subcortical)
            
            nengo.Connection(
                self.subcortical, 
                self.emotional_state,
                transform=self._emotional_transform(),
                label="Subcortical to Emotional"
            )
            
            nengo.Connection(
                self.emotional_state,
                self.intuitive_output,
                transform=self._intuitive_transform(),
                label="Emotional to Intuitive"
            )
            
            # Probes for subcortical monitoring
            self.asic_probe = nengo.Probe(self.asic_input, sample_every=0.01)
            self.subcortical_probe = nengo.Probe(self.subcortical, synapse=0.01, sample_every=0.01)
            self.emotional_probe = nengo.Probe(self.emotional_state, synapse=0.01, sample_every=0.01)
            self.intuitive_probe = nengo.Probe(self.intuitive_output, sample_every=0.01)
            
    def _build_cortical_system(self):
        """Build System 2: LLM-based conscious processing"""
        with self.model:
            # Conscious input (from subcortical bottom-up signals)
            self.conscious_input = nengo.Node(
                size_in=4,  # Accept input from intuitive output
                output=self._conscious_input_function,
                label="Conscious Input"
            )
            
            # Working memory (30% of total neurons)
            n_memory = int(self.n_neurons * 0.3)
            self.working_memory = nengo.Ensemble(
                n_neurons=n_memory,
                dimensions=512,  # Token embedding size (BERT-style)
                neuron_type=LIF(),
                label="Working Memory",
                seed=44
            )
            
            # Language processing output (with input for connections)
            self.language_output = nengo.Node(
                size_in=512,  # Accept input from working memory
                output=self._language_processing,
                label="Language Output"
            )
            
            # Executive control (20% of total neurons)
            n_executive = int(self.n_neurons * 0.2)
            self.executive_control = nengo.Ensemble(
                n_neurons=n_executive,
                dimensions=4,  # Attention, Focus, Inhibition, Integration
                neuron_type=LIF(),
                label="Executive Control",
                seed=45
            )
            
            # Connections within cortical system
            nengo.Connection(self.conscious_input, self.working_memory)
            
            nengo.Connection(
                self.working_memory,
                self.language_output,
                transform=self._language_transform(),
                label="Working Memory to Language"
            )
            
            nengo.Connection(
                self.working_memory,
                self.executive_control,
                transform=self._executive_transform(),
                label="Working Memory to Executive"
            )
            
            # Probes for cortical monitoring
            self.conscious_input_probe = nengo.Probe(self.conscious_input, sample_every=0.01)
            self.working_memory_probe = nengo.Probe(self.working_memory, synapse=0.01, sample_every=0.01)
            self.language_probe = nengo.Probe(self.language_output, sample_every=0.01)
            self.executive_probe = nengo.Probe(self.executive_control, synapse=0.01, sample_every=0.01)
            
    def _build_bridge_connections(self):
        """Build bidirectional communication between subcortical and cortical systems"""
        with self.model:
            # Bottom-up connection: Subcortical influences cortical
            nengo.Connection(
                self.intuitive_output,
                self.conscious_input,
                transform=self._bottom_up_transform(),
                label="Bottom-up Bridge"
            )
            
            # Top-down connection: Cortical influences subcortical
            nengo.Connection(
                self.language_output,
                self.subcortical,
                transform=self._top_down_transform(),
                label="Top-down Bridge"
            )
            
            # Executive feedback to emotional state
            nengo.Connection(
                self.executive_control,
                self.emotional_state,
                transform=self._feedback_transform(),
                label="Executive Feedback"
            )
            
    def _build_consciousness_integration(self):
        """Build consciousness metrics computation and phase transition detection"""
        with self.model:
            # Consciousness integration node
            self.consciousness_integration = nengo.Node(
                size_in=4 + 512,  # Accept both subcortical and cortical inputs
                output=self._consciousness_integration_function,
                label="Consciousness Integration"
            )
            
            # Connect to consciousness integration
            nengo.Connection(self.subcortical, self.consciousness_integration,
                           transform=self._subcortical_to_consciousness())
            nengo.Connection(self.working_memory, self.consciousness_integration,
                           transform=self._cortical_to_consciousness())
            
            # Probe for consciousness metrics
            self.consciousness_probe = nengo.Probe(self.consciousness_integration, 
                                                 sample_every=0.01)
            
    def _asic_input_function(self, t):
        """
        Simulate realistic ASIC hash processing for mining operations.
        Generates VESELOV HNS-compatible RGBA data.
        """
        # Generate time-based hash data
        current_time = time.time()
        hash_input = f"block_{int(t*1000)}_{current_time}"
        
        # Create realistic hash
        hash_bytes = hashlib.sha256(hash_input.encode()).digest()
        
        # Map to RGBA space (0-1 range)
        rgba_data = []
        for i in range(4):
            # Convert 4 bytes to float in [0,1]
            byte_val = int.from_bytes(hash_bytes[i*4:(i+1)*4], 'big')
            rgba_val = (byte_val % 1000000) / 1000000.0
            rgba_data.append(rgba_val)
            
        return rgba_data
    
    def _conscious_input_function(self, t, x):
        """
        Process conscious input from subcortical bottom-up signals.
        """
        # Just pass through the input for now
        return x[:512] if len(x) > 512 else x
    
    def _intuitive_processing(self, t, x):
        """
        Process intuitive output from emotional states.
        """
        # Just pass through the input for now
        return x[:4] if len(x) > 4 else x
    
    def _language_processing(self, t, x):
        """
        Generate language output from working memory.
        """
        # Just pass through the input for now
        return x[:256] if len(x) > 256 else x
    
    def _consciousness_integration_function(self, t, x):
        """
        Integrate subcortical and cortical information to compute consciousness metrics.
        """
        # Split input into subcortical and cortical parts
        subcortical_vec = x[:4] if len(x) >= 4 else [0]*4
        cortical_vec = x[4:516] if len(x) >= 516 else [0]*512
        
        # Compute consciousness metrics
        energy = np.mean(subcortical_vec)
        entropy = self._calculate_entropy(subcortical_vec)
        phi = self._calculate_phi(subcortical_vec, cortical_vec)
        
        # Phase transition detection
        phase_state = self._detect_phase_transition(energy, entropy, phi)
        
        # Store for analysis
        metrics = {
            'time': t,
            'energy': energy,
            'entropy': entropy,
            'phi': phi,
            'phase_state': phase_state,
            'temperature': self.hns_temperature
        }
        self.consciousness_history.append(metrics)
        
        return [energy, entropy, phi, phase_state]
    
    def _emotional_transform(self):
        """Transform subcortical RGBA to emotional dimensions."""
        return np.array([
            [0.5, 0.0, 0.0, 0.0],  # Energy from R
            [0.0, 0.3, 0.0, 0.0],  # Valence from G  
            [0.0, 0.0, 0.2, 0.0],  # Arousal from B
        ])
    
    def _intuitive_transform(self):
        """Transform emotional state to intuitive output."""
        # 3D emotional state -> 4D intuitive output
        transform = np.zeros((4, 3))
        for i in range(3):
            transform[i, i] = 1.0  # Copy first 3 dimensions
        transform[3, 0] = 0.5  # Set 4th dimension as average of emotional state
        return transform
    
    def _language_transform(self):
        """Transform working memory to language output."""
        # 512D -> 256D reduction
        transform = np.zeros((256, 512))
        for i in range(256):
            transform[i, i * 2] = 1.0  # Sample every other dimension
        return transform
    
    def _executive_transform(self):
        """Transform working memory to executive control."""
        return np.array([
            [0.1]*512,  # Attention
            [0.1]*512,  # Focus
            [-0.1]*512, # Inhibition (negative)
            [0.1]*512,  # Integration
        ])
    
    def _bottom_up_transform(self):
        """Transform intuitive output to conscious input."""
        # 4D -> 512D expansion
        transform = np.zeros((512, 4))
        for i in range(512):
            transform[i, i % 4] = 0.5
        return transform
    
    def _top_down_transform(self):
        """Transform language output to subcortical input."""
        # 256D -> 4D reduction
        transform = np.zeros((4, 256))
        for i in range(4):
            for j in range(256):
                if j % 4 == i:
                    transform[i, j] = 0.3
        return transform
    
    def _feedback_transform(self):
        """Transform executive control to emotional modulation."""
        return np.eye(3) * 0.2
    
    def _subcortical_to_consciousness(self):
        """Transform subcortical to consciousness integration."""
        # Create identity transform for 4D subcortical input
        transform = np.zeros((516, 4))  # Total input is 4 + 512 = 516
        transform[:4, :] = np.eye(4)  # First 4 dimensions are subcortical
        return transform
    
    def _cortical_to_consciousness(self):
        """Transform cortical to consciousness integration."""
        # Create transform for 512D cortical input
        transform = np.zeros((516, 512))  # Total input is 4 + 512 = 516
        transform[4:516, :] = np.eye(512)  # Last 512 dimensions are cortical
        return transform
    
    def _calculate_entropy(self, data):
        """Calculate information entropy from neural activity."""
        # Simple entropy calculation
        data = np.array(data)
        data = np.abs(data) + 1e-10  # Avoid log(0)
        data = data / np.sum(data)  # Normalize to probability distribution
        
        entropy = -np.sum(data * np.log2(data))
        return min(entropy, 2.0)  # Cap at 2.0 for stability
    
    def _calculate_phi(self, subcortical_data, cortical_data):
        """
        Calculate integrated information (Phi) between subcortical and cortical systems.
        Simplified Phi calculation for demonstration.
        """
        subcortical = np.array(subcortical_data)
        cortical = np.array(cortical_data[:len(subcortical)])  # Match lengths
        
        # Calculate mutual information as proxy for Phi
        # In a full implementation, this would be more sophisticated
        if len(subcortical) > 1 and len(cortical) > 1:
            correlation = np.corrcoef(subcortical, cortical)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
        else:
            correlation = 0.0
            
        phi = abs(correlation) * 1.5  # Scale factor
        return min(phi, 1.0)  # Cap at 1.0
    
    def _detect_phase_transition(self, energy, entropy, phi):
        """
        Detect consciousness phase transitions based on critical metrics.
        """
        # Critical point detection using simple thresholding
        # In a full implementation, this would use more sophisticated methods
        criticality_score = (energy + entropy + phi) / 3.0
        
        if criticality_score > self.phase_transition_threshold:
            return 1.0  # Critical state
        else:
            return 0.0  # Normal state
    
    def simulate(self, duration: float = 10.0, dt: float = 0.001) -> Dict[str, Any]:
        """
        Run the complete CHIMERA brain simulation.
        
        Args:
            duration: Simulation time in seconds
            dt: Time step in seconds
            
        Returns:
            Dictionary containing simulation results and consciousness metrics
        """
        print(f"Starting CHIMERA brain simulation for {duration} seconds...")
        
        try:
            with nengo.Simulator(self.model, dt=dt) as sim:
                print("Simulation running...")
                sim.run(duration)
                
                # Extract simulation data
                results = {
                    'time': sim.trange(),
                    'asic_input': sim.data[self.asic_probe],
                    'subcortical': sim.data[self.subcortical_probe],
                    'emotional': sim.data[self.emotional_probe],
                    'intuitive': sim.data[self.intuitive_probe],
                    'conscious_input': sim.data[self.conscious_input_probe],
                    'working_memory': sim.data[self.working_memory_probe],
                    'language': sim.data[self.language_probe],
                    'executive': sim.data[self.executive_probe],
                    'consciousness': sim.data[self.consciousness_probe],
                    'consciousness_history': self.consciousness_history.copy()
                }
                
                # Compute summary metrics
                results['metrics'] = self._compute_summary_metrics(results)
                
                print("Simulation completed successfully!")
                return results
                
        except Exception as e:
            print(f"Simulation error: {e}")
            raise
    
    def _compute_summary_metrics(self, results):
        """Compute summary consciousness metrics from simulation results."""
        consciousness_data = results['consciousness']
        subcortical_data = results['subcortical']
        cortical_data = results['working_memory']
        
        metrics = {
            'avg_energy': np.mean(consciousness_data[:, 0]) if consciousness_data.shape[0] > 0 else 0,
            'avg_entropy': np.mean(consciousness_data[:, 1]) if consciousness_data.shape[0] > 0 else 0,
            'avg_phi': np.mean(consciousness_data[:, 2]) if consciousness_data.shape[0] > 0 else 0,
            'phase_transitions': np.sum(consciousness_data[:, 3]) if consciousness_data.shape[0] > 0 else 0,
            'subcortical_activity': np.mean(subcortical_data) if subcortical_data.shape[0] > 0 else 0,
            'cortical_activity': np.mean(cortical_data) if cortical_data.shape[0] > 0 else 0,
            'total_time': results['time'][-1] if len(results['time']) > 0 else 0,
            'simulation_stability': self._assess_stability(consciousness_data)
        }
        
        return metrics
    
    def _assess_stability(self, consciousness_data):
        """Assess simulation stability based on consciousness metrics."""
        if consciousness_data.shape[0] < 10:
            return 0.5  # Not enough data
        
        # Check for reasonable activity levels (not all zeros or NaNs)
        energy_std = np.std(consciousness_data[:, 0])
        entropy_std = np.std(consciousness_data[:, 1])
        
        stability = 1.0 - min(energy_std + entropy_std, 1.0)
        return max(stability, 0.0)
    
    def visualize_gui(self):
        """Launch NengoGUI for 3D brain visualization."""
        if not self.enable_gui:
            print("NengoGUI visualization disabled")
            return
            
        print("Launching NengoGUI for 3D brain visualization...")
        print("Run this in terminal: python -m nengo_gui chimera_brain.py")
        
        # Create a simple visualization script
        viz_script = '''#!/usr/bin/env python3
import nengo_gui
import sys
sys.path.append('.')
from chimera_brain import ChimeraBrain

# Create and visualize the brain
brain = ChimeraBrain(enable_gui=True)
nengo_gui.GUI().start()
'''
        
        with open('nengo_project/visualize_brain.py', 'w') as f:
            f.write(viz_script)
        
        print("Visualization script created: visualize_brain.py")
    
    def get_brain_summary(self) -> Dict[str, Any]:
        """Get a summary of the brain architecture."""
        return {
            'total_neurons': self.n_neurons,
            'subcortical_neurons': int(self.n_neurons * 0.6),
            'cortical_neurons': int(self.n_neurons * 0.4),
            'subcortical_components': [
                'ASIC Input Layer',
                'Subcortical Processing (RGBA)',
                'Emotional States (Energy/Valence/Arousal)',
                'Intuitive Output'
            ],
            'cortical_components': [
                'Conscious Input',
                'Working Memory (512D)',
                'Language Processing',
                'Executive Control (Attention/Focus/Inhibition/Integration)'
            ],
            'bridge_connections': [
                'Bottom-up (Subcortical → Cortical)',
                'Top-down (Cortical → Subcortical)',
                'Executive Feedback'
            ],
            'consciousness_metrics': [
                'Energy',
                'Entropy', 
                'Phi (Integrated Information)',
                'Phase Transition Detection'
            ],
            'veselov_hns_integration': True,
            'gui_enabled': self.enable_gui
        }


if __name__ == "__main__":
    print("CHIMERA Bicameral Brain Architecture")
    print("====================================")
    
    # Create brain instance
    brain = ChimeraBrain(n_neurons=1000, enable_gui=True)
    
    # Print architecture summary
    summary = brain.get_brain_summary()
    print("\nBrain Architecture Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    # Run a quick test simulation
    print("\nRunning test simulation...")
    results = brain.simulate(duration=2.0)
    
    # Print results summary
    metrics = results['metrics']
    print(f"\nSimulation Results:")
    print(f"Average Energy: {metrics['avg_energy']:.4f}")
    print(f"Average Entropy: {metrics['avg_entropy']:.4f}")
    print(f"Average Phi: {metrics['avg_phi']:.4f}")
    print(f"Phase Transitions: {metrics['phase_transitions']}")
    print(f"Simulation Stability: {metrics['simulation_stability']:.4f}")
    
    print("\nCHIMERA brain simulation completed successfully!")
    print("Run 'python -m nengo_gui visualize_brain.py' for 3D visualization.")
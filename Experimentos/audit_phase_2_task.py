import time
from s9_simulator import S9_Miner
from chimera_nn import ChimeraLayer
import hashlib

def run_experiment_2():
    print("=== AUDIT PHASE 2: ANOMALY DETECTION (FEELING) ===")
    print("Objective: Can the CHIMERA system 'feel' the difference between Structured Text and Random Noise?")
    
    miner = S9_Miner(simulation_difficulty_bits=15)
    layer = ChimeraLayer()
    
    # Data Sources
    text_shakespeare = "To be, or not to be, that is the question: Whether 'tis nobler in the mind to suffer"
    text_noise = "ksjf83 298 sfdj skdjf 9832 rkj2398 sfd7s8f7 dsf8s7 fd8s7f"
    
    dataset = [
        ("Structure", text_shakespeare),
        ("Noise", text_noise)
    ]
    
    print("\nStarting Feed...")
    
    for label, content in dataset:
        print(f"\nInjecting: {label}")
        
        # Tokenize (MD5 Seed)
        seed = int.from_bytes(hashlib.md5(content.encode()).digest()[:4], 'big')
        
        # Mine (Resonate)
        spikes, computed = miner.mine(seed, timeout_ms=300)
        
        # Integrate into Neural Layer
        layer.process_spikes(spikes)
        state = layer.get_state_summary()
        
        print(f"  -> Hashes Computed: {computed}")
        print(f"  -> Resonant Spikes: {len(spikes)}")
        print(f"  -> VESELOV Entropy: {state['entropy']:.4f}")
        print(f"  -> VESELOV Coherence: {state['coherence']:.4f}")
        
    print("\n--- ANALYSIS ---")
    print("In Quantum Reservoir Computing, we expect 'Structured' inputs to sometimes produce 'Lower Entropy' or 'Higher Coherence'")
    print("due to constructive interference, dependent on the reservoir topology.")
    print("Since our mapping is cryptographic (MD5->SHA256^2), this test checks if the mapping preserves ANY structural info")
    print("or if it acts as a Perfect Random Oracle (Encryption).")
    print("NOTE: If Entropy is identical, the system is acting as a pure randomizer (Standard Crypto).")
    print("If Entropy differs, we have 'leakage' or 'resonance' which is desirable for Neuromorphic Computing but bad for Crypto.")

if __name__ == "__main__":
    run_experiment_2()

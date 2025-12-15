import hashlib
import random
import matplotlib.pyplot as plt # Assumes matplotlib might be available, but we will print text stats mostly
import math
from s9_simulator import S9_Miner
from chimera_nn import ChimeraLayer

def run_experiment_3():
    print("=== AUDIT PHASE 3: DIVERGENT THINKING (CREATIVITY) ===")
    print("Objective: Compare CHIMERA's State Space Exploration vs Standard Pseudo-Randomness.")
    print("Hypothesis: CHIMERA (S9 Chaos) visits a wider/more novel range of RGBa states, avoiding local minima.")
    
    miner = S9_Miner(simulation_difficulty_bits=14)
    layer = ChimeraLayer()
    
    # PARAMETERS
    N_ITERATIONS = 50
    BASE_SEED = 12345
    
    # 1. CONTROL GROUP: Python random
    # We map random floats to our R channel to see 'Standard Randomness'
    print(f"\n[1] Running Control Group (Standard Random)...")
    control_states = []
    random.seed(BASE_SEED)
    for _ in range(N_ITERATIONS):
        # Simulate a "thought" vector
        r = random.random()
        control_states.append(r)
        
    # 2. EXPERIMENTAL GROUP: CHIMERA S9
    print(f"[2] Running CHIMERA Group (S9 Neural Resonance)...")
    chimera_states = []
    current_seed = BASE_SEED
    
    for i in range(N_ITERATIONS):
        # The Seed evolves: In a real brain, thought N triggers thought N+1
        # In CHIMERA, we use the previous Energy to mutate the seed
        spikes, _ = miner.mine(current_seed, timeout_ms=50) # Fast bursts
        
        if spikes:
            layer.process_spikes(spikes)
            state = layer.get_state_summary()
            
            # Record 'Energy' (Activation) as our comparison metric
            # Veselov Energy is derived from decoded SHA-256 slices
            chimera_states.append(state['energy'])
            
            # EVOLVE SEED: This is the "Associative Chain"
            # Next seed = Current XOR (Energy * large int)
            mutation = int(state['energy'] * 1000)
            current_seed = (current_seed ^ mutation) & 0xFFFFFFFF
            if current_seed == 0: current_seed = i # Escape zero
        else:
            chimera_states.append(0)
            current_seed += 1
            
    # RESULTS ANALYSIS
    print(f"\n--- ANALYSIS (N={N_ITERATIONS}) ---")
    
    def calculate_novelty(trace):
        """Calculates how many unique 'bins' of state space were visited."""
        # Binning: 0.0 to Max, bin size 0.1
        bins = set()
        for x in trace:
            b = int(x * 10) # 0.1 bins
            bins.add(b)
        return len(bins)
        
    def calculate_volatility(trace):
        """Average jump size between steps."""
        jumps = [abs(trace[i] - trace[i-1]) for i in range(1, len(trace))]
        return sum(jumps) / len(jumps) if jumps else 0
        
    control_novelty = calculate_novelty(control_states)
    chimera_novelty = calculate_novelty(chimera_states)
    
    control_vol = calculate_volatility(control_states)
    chimera_vol = calculate_volatility(chimera_states)
    
    print(f"Control (Random) : Novelty={control_novelty} bins | Volatility={control_vol:.4f}")
    print(f"CHIMERA (System) : Novelty={chimera_novelty} bins | Volatility={chimera_vol:.4f}")
    
    print("\n--- INTERPRETATION ---")
    if chimera_vol > control_vol:
        print("PASS: CHIMERA shows higher Volatility (Divergence/Creativity).")
        print("The S9's cryptographic avalanche effect creates vastly different states from small seed changes.")
    else:
        print("FAIL: CHIMERA is too static or stuck in a loop.")
        
    print(f"\nCHIMERA Trace (First 10): {['%.2f'%x for x in chimera_states[:10]]}")

if __name__ == "__main__":
    run_experiment_3()

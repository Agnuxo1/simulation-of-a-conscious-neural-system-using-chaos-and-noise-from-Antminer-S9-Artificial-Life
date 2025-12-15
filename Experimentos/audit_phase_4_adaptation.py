import hashlib
import matplotlib.pyplot as plt
from s9_simulator import S9_Miner
from chimera_nn import ChimeraLayer

def run_experiment_4():
    print("=== AUDIT PHASE 4: NEUROMORPHIC ADAPTATION (STDP) ===")
    print("Objective: Verify that Repeated Stimuli cause changes in Synaptic Weight (Learning).")
    
    miner = S9_Miner(simulation_difficulty_bits=15)
    layer = ChimeraLayer()
    
    # Define a Stimulus
    stimulus_text = "Pain conditioning" 
    seed = int.from_bytes(hashlib.md5(stimulus_text.encode()).digest()[:4], 'big')
    
    print(f"Stimulus: '{stimulus_text}' (Seed: {seed})")
    
    energy_trace = []
    weight_trace = []
    
    # Repetition Loop
    print("Presenting stimulus 20 times...")
    for i in range(20):
        # 1. Mine (Resonate)
        spikes, _ = miner.mine(seed, timeout_ms=200)
        
        # 2. Process with STDP (Pass context_key=seed)
        layer.process_spikes(spikes, context_key=seed)
        
        # 3. Record State
        state = layer.get_state_summary()
        energy_trace.append(state['energy'])
        
        # Access internal weight for audit
        current_weight = layer.synaptic_weights.get(seed, 1.0)
        weight_trace.append(current_weight)
        
        print(f"  Run {i+1}: Weight={current_weight:.3f} | Energy={state['energy']:.2f}")

    print("\n--- ANALYSIS ---")
    start_w = weight_trace[0]
    end_w = weight_trace[-1]
    
    print(f"Initial Weight: {start_w:.3f}")
    print(f"Final Weight:   {end_w:.3f}")
    
    change_pct = ((end_w - start_w) / start_w) * 100
    print(f"Plasticity Delta: {change_pct:.2f}%")
    
    if abs(change_pct) > 1.0:
        if change_pct > 0:
            print("RESULT: SENSITIZATION. The system learned to amplify this signal (LTP).")
        else:
            print("RESULT: HABITUATION. The system learned to ignore this signal (LTD).")
        print("PASS: STDP is active.")
    else:
        print("FAIL: No significant adaptation occurred. (Check 'B' channel levels)")

if __name__ == "__main__":
    run_experiment_4()

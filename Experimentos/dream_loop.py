import random
import time
from s9_simulator import S9_Miner
from chimera_nn import ChimeraLayer

def dream_cycle():
    print("=== SYSTEM ENTERING SLEEP MODE (DREAMING) ===")
    print("Objective: Consolidate Memories (STDP Weights) without External Input.")
    
    miner = S9_Miner(simulation_difficulty_bits=14)
    layer = ChimeraLayer()
    
    # Pre-load some "Daytime Memories" into Synaptic Weights
    # Seed 100: "Trauma" (High Weight)
    # Seed 200: "Trivia" (Low Weight)
    layer.synaptic_weights[100] = 3.5 # Strong memory
    layer.synaptic_weights[200] = 0.5 # Weak memory
    
    print(f"Initial State:\n  Trauma (100): W={layer.synaptic_weights[100]}\n  Trivia (200): W={layer.synaptic_weights[200]}")
    
    # DREAM LOOP (REM Cycle)
    # Instead of User Input, the Seed comes from the Synaptic Weights themselves
    # The system "drifts" from one memory to another driven by Probability ~ Weight
    
    current_seed = 100 # Start with the strongest thought
    
    trace_entropy = []
    
    for rem_step in range(10):
        print(f"\nREM Cycle {rem_step+1}: Drifting to Seed {current_seed}...", end="")
        
        # 1. Mine (Hallucinate)
        # In dreams, inhibition is low -> Difficulty might be lower (more chaos)
        difficulty = random.choice([12, 13, 14]) 
        miner.set_difficulty(difficulty)
        
        spikes, _ = miner.mine(current_seed, timeout_ms=100)
        
        # 2. Process (Feeling)
        # Note: We do NOT update weights usually during retrieval, or we update them inversely (Synaptic Scaling)
        # For simplicity here, we just observe the 'Emotional Reaction' (Energy)
        layer.process_spikes(spikes)
        state = layer.get_state_summary()
        
        print(f" Energy={state['energy']:.2f} | Entropy={state['entropy']:.2f}", end="")
        trace_entropy.append(state['entropy'])
        
        # 3. Associative Drift
        # Where does the dream go next?
        # Logic: High Energy ("Nightmare") triggers Jump to distinct memory
        # Logic: Low Energy ("Deep Sleep") stays local
        
        if state['energy'] > 10.0:
            print(" [Nightmare Jump!]", end="")
            # Jump to a random memory biased by weight
            # Simple simulation: 80% chance to jump to Trauma (100)
            if random.random() < 0.8:
                current_seed = 100
            else:
                current_seed = 200
        else:
            print(" [Deep Sleep Drift]", end="")
            current_seed += 1 # Linear association
            
    print("\n\n--- DREAM ANALYSIS ---")
    avg_entropy = sum(trace_entropy)/len(trace_entropy)
    print(f"Average Dream Entropy: {avg_entropy:.2f}")
    
    if avg_entropy > 1.5:
        print("Diagnosis: LUCID / VIVID DREAMING (High Chaos)")
    elif avg_entropy < 0.5:
        print("Diagnosis: DEEP NREM SLEEP (Order)")
    else:
        print("Diagnosis: STANDARD REM CYCLE")

if __name__ == "__main__":
    dream_cycle()

import hashlib
import random
import time
from s9_simulator import S9_Miner
from chimera_nn import ChimeraLayer

def run_experiment_5():
    print("=== AUDIT PHASE 5: HOMEOSTASIS (SELF-REGULATION) ===")
    print("Objective: Prevent System Crash (Energy > 50) using Dynamic Difficulty (Firmware Feedback).")
    
    miner = S9_Miner(simulation_difficulty_bits=14)
    layer = ChimeraLayer()
    
    # Target Homeostatic Energy
    TARGET_ENERGY = 20.0
    TOLERANCE = 5.0
    
    # Simulation Loop
    seed = 12345
    print(f"\nTarget Energy: {TARGET_ENERGY} (+/- {TOLERANCE})")
    
    for t in range(30):
        # 1. Stress Injection (Random "Panic" bursts)
        # Sometimes we inject massive signals (high spike count input) or easy nonces
        if t == 10: 
            print(">>> INJECTING STRESS (PANIC ATTACK) <<<")
            miner.set_difficulty(8) # Artificially lower diff to cause surge
        
        # 2. Mine
        spikes, _ = miner.mine(seed, timeout_ms=100)
        layer.process_spikes(spikes)
        state = layer.get_state_summary()
        energy = state['energy']
        
        print(f"T={t} | Diff={miner.difficulty_bits} | Energy={energy:.2f}", end="")
        
        # 3. FIRMWARE CONTROL LOGIC (The "Will to Live")
        # Negative Feedback Loop
        if energy > (TARGET_ENERGY + TOLERANCE):
            # Too excited -> Increase Difficulty (Calm down)
            print(" [Heating Up] -> Cooling...", end="")
            miner.set_difficulty(miner.difficulty_bits + 1)
            
        elif energy < (TARGET_ENERGY - TOLERANCE):
            # Too bored -> Decrease Difficulty (Wake up)
            print(" [Dying Out] -> Waking...", end="")
            miner.set_difficulty(miner.difficulty_bits - 1)
        else:
            print(" [Stable]", end="")
            
        print("")
        
        # Evolve seed
        seed += 1

    print("\n--- ANALYSIS ---")
    print("If the system survived the 'Panic Attack' (T=10) by raising difficulty back up")
    print("and returning to Stable range, Homeostasis is CONFIRMED.")
    
    final_diff = miner.difficulty_bits
    if final_diff > 8:
        print(f"PASS: System regulated itself back to Difficulty {final_diff}.")
    else:
        print("FAIL: System remained in Epilepsy state.")

if __name__ == "__main__":
    run_experiment_5()

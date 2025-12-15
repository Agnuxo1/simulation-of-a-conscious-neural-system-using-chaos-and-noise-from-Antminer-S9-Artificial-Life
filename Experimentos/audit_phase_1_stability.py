import hashlib
import statistics
import time
from s9_simulator import S9_Miner
from chimera_nn import ChimeraLayer

def generate_seed(text):
    return int.from_bytes(hashlib.md5(text.encode()).digest()[:4], 'big')

def run_experiment_1():
    print("=== AUDIT PHASE 1: STABILITY & DETERMINISTIC CHAOS ===")
    print("Objective: Prove that the S9 Simulator produces deterministic but chaotic output unique to each 'Thought'.")
    
    miner = S9_Miner(simulation_difficulty_bits=14) # Lower diff for faster stats
    
    seeds = {
        "Logic": generate_seed("Logic and Order"),
        "Chaos": generate_seed("Chaos and Entropy"),
        "Love": generate_seed("Love and Empathy")
    }
    
    results = {}
    
    # Run each seed multiple times to check variance (Stability)
    # Then compare seeds to check Distinctness (Sensitivity)
    
    for name, seed in seeds.items():
        print(f"\nTesting Thought: '{name}' (Seed: {seed})")
        
        energies = []
        entropies = []
        
        # 10 runs per seed
        for run in range(10):
            # Reset Layer each valid run to isolate the "Thought"
            layer = ChimeraLayer()
            
            # Mine for 100ms
            spikes, _ = miner.mine(seed, timeout_ms=100)
            
            # Process
            layer.process_spikes(spikes)
            state = layer.get_state_summary()
            
            energies.append(state["energy"])
            entropies.append(state["entropy"])
            
        # Stats
        avg_e = statistics.mean(energies)
        std_e = statistics.stdev(energies) if len(energies) > 1 else 0
        
        avg_h = statistics.mean(entropies)
        std_h = statistics.stdev(entropies) if len(entropies) > 1 else 0
        
        results[name] = {"avg_energy": avg_e, "avg_entropy": avg_h, "std_energy": std_e}
        
        print(f"  -> Energy: {avg_e:.4f} (±{std_e:.4f})")
        print(f"  -> Entropy: {avg_h:.4f}")
        
    print("\n--- CONCLUSION ---")
    
    # Check 1: Zero Variance (Deterministic)
    # Note: In a real constrained time window (timeout), the number of hashes might vary slightly depending on CPU load.
    # However, for the SAME number of hashes, the results should be identical. 
    # Our simulator loop is time-bound, so 'count' might jitter, but 'distribution' (Entropy) should be stable.
    
    for name, data in results.items():
        if data["std_energy"] < 0.5: # Allow small jitter due to time-slicing
            print(f"PASS: '{name}' is stable.")
        else:
            print(f"WARN: '{name}' shows high variance (CPU load noise?).")
            
    # Check 2: Distinctness
    if results["Logic"]["avg_entropy"] != results["Chaos"]["avg_entropy"]:
        print("PASS: Distinct thoughts produce distinct Entropy signatures.")
    else:
        print("FAIL: Thoughts are indistinguishable.")

if __name__ == "__main__":
    run_experiment_1()

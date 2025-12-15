import hashlib
import random
import time
import matplotlib.pyplot as plt
from s9_simulator import S9_Miner
from chimera_nn import ChimeraLayer

def generate_market_tick(tick_type="NORMAL"):
    """
    Simulates a market trade.
    NORMAL: Pure random variation.
    MANIPULATED: Algorithmic pattern (repeating seeds).
    """
    if tick_type == "NORMAL":
        # Random unique trade ID
        return f"TRADE_{random.randint(0, 100000000)}_{time.time()}"
    else:
        # Bot network executing specific orders (Repeating patterns)
        # 10 specific signatures
        pattern_id = random.randint(0, 5) 
        return f"ALGO_ORDER_66_{pattern_id}"

def run_sentinel():
    print("=== CHIMERA PHASE 14: THE SENTINEL ===")
    print("Objective: Detect 'Algorithmic Manipulation' (Hidden Patterns) in Market Stream.")
    print("Mechanism: STDP Sensitization (The network 'heats up' on repeating patterns).")
    
    miner = S9_Miner(simulation_difficulty_bits=14)
    layer = ChimeraLayer()
    
    energy_trace = []
    
    print("\n[Phase 1] Monitoring Organic Market Traffic (0-50 ticks)...")
    
    # 1. Normal Traffic
    # Weights should stay near 1.0 because input is random/novel
    for i in range(50):
        tick_data = generate_market_tick("NORMAL")
        seed = int.from_bytes(hashlib.md5(tick_data.encode()).digest()[:4], 'big')
        
        spikes, _ = miner.mine(seed, timeout_ms=30)
        layer.process_spikes(spikes, context_key=seed)
        state = layer.get_state_summary()
        energy_trace.append(state['energy'])
        
        if i % 10 == 0:
            print(f"  Tick {i}: Energy={state['energy']:.2f} (Baseline)")

    print("\n[Phase 2] INJECTION: High-Frequency Algo Attack (50-100 ticks)...")
    
    # 2. Manipulated Traffic
    # Repeating patterns should trigger STDP Sensitization (LTP)
    # Weights for these specific seeds will rise
    # Energy = Activation * Weight -> Energy should SPIKE
    
    algo_start_energy = 0
    
    for i in range(50, 100):
        tick_data = generate_market_tick("MANIPULATED")
        seed = int.from_bytes(hashlib.md5(tick_data.encode()).digest()[:4], 'big')
        
        spikes, _ = miner.mine(seed, timeout_ms=30)
        layer.process_spikes(spikes, context_key=seed)
        state = layer.get_state_summary()
        energy_trace.append(state['energy'])
        
        if i == 50: algo_start_energy = state['energy']
        
        # ALERT LOGIC
        if state['energy'] > 30.0:
            print(f"  Tick {i}: Energy={state['energy']:.2f} >>> ANOMALY DETECTED (High Resonance) <<<")
        elif i % 10 == 0:
             print(f"  Tick {i}: Energy={state['energy']:.2f} (Climbing...)")
             
    print("\n--- SENTINEL REPORT ---")
    avg_normal = sum(energy_trace[:50]) / 50
    avg_attack = sum(energy_trace[50:]) / 50
    
    print(f"Average Energy (Organic): {avg_normal:.2f}")
    print(f"Average Energy (Attack):  {avg_attack:.2f}")
    
    sensitivity_gain = ((avg_attack - avg_normal) / avg_normal) * 100
    print(f"Sensitivity Gain: +{sensitivity_gain:.1f}%")
    
    if sensitivity_gain > 50.0:
        print("RESULT: SUCCESS. The Sentinel 'felt' the market manipulation.")
    else:
        print("RESULT: FAIL. Signal indistinguishable from noise.")

if __name__ == "__main__":
    run_sentinel()

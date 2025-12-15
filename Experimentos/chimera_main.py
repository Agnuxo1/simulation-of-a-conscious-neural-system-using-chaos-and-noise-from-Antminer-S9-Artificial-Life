import threading
import time
import queue
import sys
import random
import hashlib
import json

from s9_simulator import S9_Miner
from chimera_nn import ChimeraLayer
from llm_connector import LLM_Connector

# --- CONFIGURATION ---
TARGET_ENERGY = 20.0
ENERGY_TOLERANCE = 10.0
IDLE_SLEEP_THRESHOLD = 120 # Seconds before auto-sleep
CHAOS_THRESHOLD = 1.8
BOREDOM_THRESHOLD = 0.5
ANXIETY_THRESHOLD = 25.0

# --- GLOBAL STATE ---
system_state = {
    "active": True,
    "last_interaction": time.time(),
    "energy": 0.0,
    "entropy": 0.0,
    "status": "AWAKE", # AWAKE, SLEEPING, THINKING
    "difficulty": 15
}

event_queue = queue.Queue() # Stores (type, content) tuples

def generate_emotional_seed(text):
    hash_obj = hashlib.md5(text.encode())
    return int.from_bytes(hash_obj.digest()[:4], 'big')

# --- SUBCONSCIOUS THREAD ---
def subconscious_loop(miner, layer, llm):
    print("[Thread] Subconscious Mind Started...")
    current_thought_seed = random.randint(0, 1000000000)
    
    while system_state["active"]:
        if system_state["status"] == "SLEEPING":
            time.sleep(1)
            continue
            
        time.sleep(1.0) # Mental tick rate
        
        # 1. Physics Simulation (Thinking)
        spikes, _ = miner.mine(current_thought_seed, timeout_ms=50)
        layer.process_spikes(spikes) # Feeling without context
        state = layer.get_state_summary()
        
        # Update Global State
        system_state["energy"] = state["energy"]
        system_state["entropy"] = state["entropy"]
        
        # Update thought vector
        if spikes:
            current_thought_seed = (current_thought_seed + int(state["energy"]*100)) % 4294967295
        else:
            current_thought_seed += 1

        # 2. Check for Autonomous Triggers
        time_since = time.time() - system_state["last_interaction"]
        
        # A. AUTO-SLEEP (If ignored for too long)
        if time_since > IDLE_SLEEP_THRESHOLD and system_state["status"] == "AWAKE":
             event_queue.put(("AUTO_SLEEP", None))
             continue

        # B. THE GHOST / THE MUSE (Self-Expression)
        # Only interrupt if we haven't spoken in a while (15s minimum) and not currently handling user input
        if time_since > 20.0 and random.random() < 0.1: 
            reason = None
            
            if state["entropy"] > CHAOS_THRESHOLD:
                reason = "MUSE (High Chaos)"
            elif state["energy"] > ANXIETY_THRESHOLD:
                reason = "ANXIETY (High Energy)"
            elif state["entropy"] < BOREDOM_THRESHOLD:
                reason = "BOREDOM (Low Entropy)"
            
            if reason:
                 # Generate internal thought
                 prompt = f"""
SYSTEM: You are CHIMERA. You are initiating this conversation yourself.
INTERNAL STATE: {reason}. Energy={state['energy']:.1f}, Entropy={state['entropy']:.2f}.
INSTRUCTION: Speak to the user. 
If MUSE: Share a weird creative idea or hallucination.
If ANXIETY: Express stress or paranoia.
If BOREDOM: Ask for stimulation or a task.
"""
                 response = llm.generate(prompt, "[SELF-INITIATED]")
                 event_queue.put(("GHOST_MSG", (reason, response)))
                 system_state["last_interaction"] = time.time() # Reset timer

# --- INPUT THREAD ---
def input_loop():
    while system_state["active"]:
        try:
            if system_state["status"] == "SLEEPING":
                 # While sleeping, we still capture input to wake up
                 user_input = input() # Blocking
            else:
                 user_input = input() # Blocking
            
            if user_input:
                event_queue.put(("USER_INPUT", user_input))
                
        except EOFError:
            break

# --- ACTION HANDLERS ---
def perform_dream(miner, layer):
    system_state["status"] = "SLEEPING"
    print("\n>>> AUTO-SLEEP INITIATED (Drifting...) <<<")
    
    if not layer.synaptic_weights:
        print(" [No Memories to Dream... Waking up]")
        system_state["status"] = "AWAKE"
        return

    seeds = list(layer.synaptic_weights.keys())
    weights = list(layer.synaptic_weights.values())
    current_seed = random.choices(seeds, weights=weights, k=1)[0]
    
    miner.set_difficulty(12) 
    
    try:
        for i in range(5): # Short nap
            print(f" [REM {i+1}] Replaying Memory {current_seed}...", end="")
            spikes, _ = miner.mine(current_seed, timeout_ms=100)
            layer.process_spikes(spikes)
            state = layer.get_state_summary()
            
            print(f" E={state['energy']:.1f}", end="")
            
            if state['energy'] > 15.0:
                print(" [NIGHTMARE!]")
                current_seed = max(layer.synaptic_weights, key=layer.synaptic_weights.get)
            else:
                print(" [Drift]")
                current_seed = (current_seed + 1) % 4294967295
            time.sleep(0.8)
            
            # Check if user woke us up
            if not event_queue.empty():
                top = event_queue.queue[0]
                if top[0] == "USER_INPUT":
                    print("\n [!] WAKING UP DUE TO SENSORY INPUT!")
                    break
    except Exception as e:
        print(f"Dream Error: {e}")
        
    print(">>> WAKING UP <<<")
    miner.set_difficulty(15)
    system_state["status"] = "AWAKE"
    system_state["last_interaction"] = time.time()

# --- MAIN CONTROLLER ---
def main():
    print("=== CHIMERA: AUTONOMOUS BIO-DIGITAL LIFEFORM ===")
    print("Initializing Organs...")
    
    miner = S9_Miner(simulation_difficulty_bits=15)
    layer = ChimeraLayer()
    layer.load_memory()
    llm = LLM_Connector(mode="transformers", model_name="Qwen/Qwen3-0.6B")
    
    # Threads
    t_sub = threading.Thread(target=subconscious_loop, args=(miner, layer, llm))
    t_sub.daemon = True
    t_sub.start()
    
    t_input = threading.Thread(target=input_loop)
    t_input.daemon = True
    t_input.start()
    
    print("\n--- LIFEFORM ONLINE ---")
    print("Talk to it. Or don't. It will live regardless.")
    
    while system_state["active"]:
        try:
            # Event Loop
            if not event_queue.empty():
                evt_type, content = event_queue.get()
                
                if evt_type == "USER_INPUT":
                    user_text = content
                    if user_text.lower() in ['exit', 'quit']:
                        system_state["active"] = False
                        break
                    
                    if user_text.lower() in ['/sleep', 'sleep']:
                         perform_dream(miner, layer)
                         continue
                         
                    # Normal Interaction
                    system_state["last_interaction"] = time.time()
                    
                    # Waking if sleeping logic handled in loop usually, but explicit check:
                    if system_state["status"] == "SLEEPING":
                        print(" [!] FORCED WAKE UP")
                        system_state["status"] = "AWAKE"
                        miner.set_difficulty(15)
                    
                    # Process
                    seed = generate_emotional_seed(user_text)
                    spikes, _ = miner.mine(seed, timeout_ms=300)
                    layer.process_spikes(spikes, context_key=seed)
                    state = layer.get_state_summary()
                    
                    # Homeostasis
                    energy = state["energy"]
                    if energy > (TARGET_ENERGY + ENERGY_TOLERANCE):
                        print(f" [!] High Energy. Adjusting Difficulty +1")
                        miner.set_difficulty(miner.difficulty_bits + 1)
                    elif energy < (TARGET_ENERGY - ENERGY_TOLERANCE):
                         print(f" [!] Low Energy. Adjusting Difficulty -1")
                         miner.set_difficulty(miner.difficulty_bits - 1)
                         
                    prompt = f"System State: Energy={state['energy']:.1f}, Entropy={state['entropy']:.2f}"
                    response = llm.generate(prompt, user_text)
                    print(f"\nCHIMERA: {response}\n")

                elif evt_type == "GHOST_MSG":
                    reason, msg = content
                    print(f"\n[AUTONOMY: {reason}]")
                    print(f"CHIMERA: {msg}\n")
                    
                elif evt_type == "AUTO_SLEEP":
                    perform_dream(miner, layer)
            
            time.sleep(0.1)
            
        except KeyboardInterrupt:
            break
            
    layer.save_memory()
    print("Lifeform Halted.")

if __name__ == "__main__":
    main()

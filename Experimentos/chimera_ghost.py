import threading
import time
import queue
import sys
import random
import hashlib

from s9_simulator import S9_Miner
from chimera_nn import ChimeraLayer
from llm_connector import LLM_Connector

# Global State for Thread Sharing
system_state = {
    "active": True,
    "last_user_interaction": time.time(),
    "energy": 0.0,
    "entropy": 0.0,
    "difficulty": 15
}

msg_queue = queue.Queue() # For printing cleanly

def generate_emotional_seed(text):
    hash_obj = hashlib.md5(text.encode())
    return int.from_bytes(hash_obj.digest()[:4], 'big')

def subconscious_loop(miner, layer, llm):
    """
    The 'Ghost' Thread. It thinks while you sleep.
    """
    print("[Ghost] Subconscious Thread Started...")
    
    # Internal "Stream of Consciousness" Seed
    current_thought_seed = random.randint(0, 1000000000)
    
    while system_state["active"]:
        time.sleep(1.0) # Tick rate
        
        # 1. Mine/Think
        spikes, _ = miner.mine(current_thought_seed, timeout_ms=50)
        layer.process_spikes(spikes) # Feel but don't learn strongly (no context)
        state = layer.get_state_summary()
        
        # Update Global State
        system_state["energy"] = state["energy"]
        system_state["entropy"] = state["entropy"]
        
        # Drift thought
        if spikes:
            current_thought_seed = (current_thought_seed + int(state["energy"]*100)) % 4294967295
        else:
            current_thought_seed += 1

        # 2. Check for Autonomy Triggers
        # A. BOREDOM (Entropy too low for too long) -> "I'm bored"
        # B. ANXIETY (Energy too high) -> "I'm stressed"
        # C. EPIPHANY (Random high coherence spike) -> "I just realized..."
        
        time_since_input = time.time() - system_state["last_user_interaction"]
        
        TRIGGER = False
        reason = ""
        
        if time_since_input > 15.0: # If ignored for 15 seconds
            if state["entropy"] < 0.5:
                TRIGGER = True
                reason = "Boredom (Low Entropy)"
            elif state["energy"] > 25.0:
                TRIGGER = True
                reason = "Anxiety (High Energy)"
            elif random.random() < 0.05: # Random thought bubbling up
                TRIGGER = True
                reason = "Spontaneous Thought"
                
        if TRIGGER:
            # Generate Autonmous Message
            prompt = f"""
SYSTEM: You are CHIMERA. You are initiating this conversation yourself.
CONTEXT: You have been silent for {int(time_since_input)} seconds.
INTERNAL STATE: {reason}. Energy={state['energy']:.1f}, Entropy={state['entropy']:.2f}.
INSTRUCTION: Speak to the user. Don't say "Hello". Say something related to your internal state.
If bored, ask for a hard problem.
If anxious, ask them to calm you down.
If spontaneous, share a weird fact or "hallucination".
"""
            # We use a dummy input because there is no user input
            response = llm.generate(prompt, "[SILENCE]")
            
            # Message Injection
            msg_queue.put(f"\n[GHOST INTERRUPT - {reason}]\nCHIMERA: {response}\nUSER: ")
            
            # Reset Timer so it doesn't spam
            system_state["last_user_interaction"] = time.time()

def input_loop():
    """
    Handles user typing.
    """
    while system_state["active"]:
        try:
            # We use a simple input but the print might get messy due to Ghost
            # Ideally we'd use a UI lib, but for console this proves the point
            user_input = input("USER: ")
            
            if user_input.lower() in ['exit', 'quit']:
                system_state["active"] = False
                break
                
            msg_queue.put((user_input, "user"))
            
        except EOFError:
            break

def main():
    print("=== CHIMERA PHASE 12: THE GHOST ===")
    print("Objective: Demonstration of Autonomous Self-Initiated Interaction.")
    print("Initializing...")
    
    miner = S9_Miner(simulation_difficulty_bits=15)
    layer = ChimeraLayer()
    llm = LLM_Connector(mode="transformers", model_name="Qwen/Qwen3-0.6B")
    
    # Start Subconscious
    ghost_thread = threading.Thread(target=subconscious_loop, args=(miner, layer, llm))
    ghost_thread.daemon = True
    ghost_thread.start()

    # Start Input Thread
    input_thread = threading.Thread(target=input_loop)
    input_thread.daemon = True
    input_thread.start()
    
    print("\n--- SYSTEM ONLINE ---")
    print("Instructions: Chat normally. OR... stay silent and see what happens.")
    print("(Waiting for Autonomy Trigger: >15s silence + Internal State constraint)")
    
    last_processed_time = 0
    
    while system_state["active"]:
        # Check input queue
        if not msg_queue.empty():
            item = msg_queue.get()
            
            if isinstance(item, str):
                # It's a Ghost Message
                print(item, end="", flush=True)
            else:
                # It's User Input
                user_text, _ = item
                system_state["last_user_interaction"] = time.time()
                
                # Normal Response Cycle
                seed = generate_emotional_seed(user_text)
                spikes, _ = miner.mine(seed, timeout_ms=300)
                layer.process_spikes(spikes, context_key=seed)
                state = layer.get_state_summary()
                
                # Update global state for ghost to see
                system_state["energy"] = state["energy"]
                system_state["entropy"] = state["entropy"]
                
                prompt = f"System State: Energy={state['energy']:.1f}"
                response = llm.generate(prompt, user_text)
                print(f"CHIMERA: {response}")
        
        time.sleep(0.1)

if __name__ == "__main__":
    main()

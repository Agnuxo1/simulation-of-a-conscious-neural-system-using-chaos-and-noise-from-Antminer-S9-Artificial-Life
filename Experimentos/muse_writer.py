import hashlib
import time
import random
from s9_simulator import S9_Miner
from chimera_nn import ChimeraLayer
from llm_connector import LLM_Connector

def analyze_chapter_emotional_impact(text, miner, layer):
    """
    "Dreams" the chapter to find its Energy signature.
    Returns: Max Energy encountered.
    """
    seed = int.from_bytes(hashlib.md5(text.encode()).digest()[:4], 'big')
    
    # Dream Cycle (Rapid Mining)
    max_energy = 0.0
    for i in range(5):
        spikes, _ = miner.mine(seed + i, timeout_ms=50) # Drift seeds
        layer.process_spikes(spikes)
        state = layer.get_state_summary()
        if state['energy'] > max_energy:
            max_energy = state['energy']
            
    return max_energy

def main():
    print("=== CHIMERA PHASE 13: THE MUSE ===")
    print("Objective: Divergent Storytelling using 'Nightmare' injections.")
    
    # Lower difficulty to 12 to guarantee "Energy" (Nightmares) for the demo
    miner = S9_Miner(simulation_difficulty_bits=12)
    layer = ChimeraLayer()
    llm = LLM_Connector(mode="transformers", model_name="Qwen/Qwen3-0.6B")
    
    premise = input("Enter Story Premise (e.g. 'A detective finds a magic watch'): ")
    story_context = f"PREMISE: {premise}\n\n"
    
    CHAOS_TOKENS = [
        "PLOT TWIST: The protagonist realizes they are a ghost.",
        "PLOT TWIST: A meteor impacts the earth immediately.",
        "PLOT TWIST: The villain reveals they are the hero's future self.",
        "PLOT TWIST: Gravity reverses.",
        "PLOT TWIST: An alien invasion starts now."
    ]
    
    for chapter_num in range(1, 4):
        print(f"\n--- WRITING CHAPTER {chapter_num} ---")
        
        # 1. Draft
        prompt = f"""
STORY SO FAR:
{story_context}
INSTRUCTION: Write Chapter {chapter_num}. Keep it concise (1 paragraph).
"""
        draft = llm.generate(prompt, "Write.")
        print(f"[Drafting]...\n{draft}\n")
        
        # 2. Dream (Critique)
        print("[Dreaming on Draft...]")
        energy = analyze_chapter_emotional_impact(draft, miner, layer)
        print(f"Emotional Energy: {energy:.2f}")
        
        # 3. Decision
        final_text = draft
        # Threshold 10.0 is easy to hit with Diff 12
        if energy > 10.0:
            print(">>> NIGHTMARE DETECTED! (High Energy) <<<")
            chaos = random.choice(CHAOS_TOKENS)
            print(f"Injecting Chaos Token: '{chaos}'")
            
            rewrite_prompt = f"""
ORIGINAL DRAFT:
{draft}
INJECTED CHAOS: {chaos}
INSTRUCTION: Rewrite the chapter interacting with the chaos token IMMEDIATELY.
"""
            final_text = llm.generate(rewrite_prompt, "Rewrite.")
            print(f"\n[REWRITTEN VERSION]:\n{final_text}")
            story_context += f"CHAPTER {chapter_num} (CHAOS INJECTED: {chaos}):\n{final_text}\n\n"
        else:
            print("[Stable Dream] -> Keeping draft.")
            story_context += f"CHAPTER {chapter_num}:\n{final_text}\n\n"
        
    print("\n=== FINAL STORY ===")
    print(story_context)
    
    with open("muse_story.txt", "w", encoding="utf-8") as f:
        f.write(story_context)
    print("Story saved to muse_story.txt")

if __name__ == "__main__":
    main()

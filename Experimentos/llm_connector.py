import requests
import json
import time

# Try importing transformers (will fail if not installed, handled gracefully)
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

class LLM_Connector:
    """
    Connects to a local LLM.
    Modes:
    1. 'ollama': Connects to local Ollama API (http://localhost:11434).
    2. 'transformers': Loads model directly in Python using HuggingFace.
    3. 'mock': Simulation if others fail.
    """
    def __init__(self, mode="transformers", model_name="Qwen/Qwen3-0.6B", api_url="http://localhost:11434/api/generate"):
        self.mode = mode
        self.model_name = model_name
        self.api_url = api_url
        self.model = None
        self.tokenizer = None
        self.connected = False
        
        print(f"[LLM] Initializing Connector. Mode={mode}, Model={model_name}")
        
        if self.mode == "transformers":
            if TRANSFORMERS_AVAILABLE:
                try:
                    print(f"[LLM] Loading model {model_name} via Transformers... (This may take a moment)")
                    self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                    # device_map="auto" requires 'accelerate' library
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name, 
                        device_map="auto", 
                        trust_remote_code=True,
                        torch_dtype="auto"
                    )
                    self.connected = True
                    print(f"[LLM] Model Loaded Successfully!")
                except ImportError as e:
                    if "accelerate" in str(e):
                        print("[LLM] ERROR: 'accelerate' library is missing. Required for device_map='auto'.")
                        print("[LLM] Please run: pip install accelerate")
                    else:
                        print(f"[LLM] Import Error: {e}")
                    print("[LLM] Falling back to Mock mode.")
                    self.mode = "mock"
                except Exception as e:
                    print(f"[LLM] Failed to load Transformers model: {e}")
                    print("[LLM] Falling back to Mock mode.")
                    self.mode = "mock"
            else:
                print("[LLM] 'transformers' library not installed. Please run: pip install transformers torch accelerate")
                print("[LLM] Falling back to Mock mode.")
                self.mode = "mock"
                
        elif self.mode == "ollama":
            # Check connection
            try:
                requests.get("http://localhost:11434/", timeout=0.2)
                self.connected = True
                print(f"[LLM] Connected to Ollama at {api_url}")
            except:
                print(f"[LLM] Ollama connection failed.")
                self.mode = "mock"

    def generate(self, system_prompt, user_input, temperature=0.7):
        """
        Generates text based on current mode.
        """
        full_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"
        
        if self.mode == "transformers" and self.connected:
            return self._generate_transformers(full_prompt, temperature)
        elif self.mode == "ollama" and self.connected:
            return self._generate_ollama(full_prompt, temperature)
        else:
            return self._mock_generate(system_prompt, user_input)

    def _generate_transformers(self, prompt, temperature):
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            # Generate
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=temperature,
                do_sample=True,
                top_p=0.9
            )
            
            # Decode (remove the prompt part)
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Simple cleanup to remove prompt echo if necessary (transformers sometimes echoes)
            # Qwen chat templates usually handle this well, but we'll return raw for now or slice
            # Identifying where assistant starts might be needed if it echoes.
            # Usually decode skip_special_tokens gives the whole conversation logic. 
            
            # Quick hack to get just the response if the model echoes:
            response = generated_text.replace(prompt.replace("<|im_start|>", "").replace("<|im_end|>", ""), "").strip()
            # If replacement didn't work perfectly due to tokenization, just return end
            return generated_text.split("assistant")[-1].strip()
            
        except Exception as e:
            return f"[Transformers Error: {e}]"

    def _generate_ollama(self, prompt, temperature):
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "temperature": temperature
            }
            response = requests.post(self.api_url, json=payload, timeout=30)
            if response.status_code == 200:
                data = response.json()
                return data.get("response", "")
            else:
                return f"[Ollama Error: {response.status_code}]"
        except Exception as e:
            return f"[Ollama Error: {e}]"

    def _mock_generate(self, system_prompt, user_input):
        time.sleep(1) # Simulate thought
        
        # Simple extraction of context for flavor
        tone = "Neutral"
        if "Energía: HIGH" in system_prompt: tone = "Excited"
        if "Entropía: HIGH" in system_prompt: tone = "Abstract/Creative"
        if "Entropía: LOW" in system_prompt: tone = "Logical/Precise"

        return f"[SIMULATED QWEN responding in {tone} tone]: I received your input '{user_input}'. My subconscious parameters suggest a {tone} approach. (Install Ollama to get real AI responses)."

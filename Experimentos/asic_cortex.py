import socket
import struct
import time
import random
import math

# Configuration HNS (NeuroCHIMERA Paper)
BASE = 1000.0 

class ASIC_Cortex:
    def __init__(self, asic_ip='192.168.1.100', mock_mode=False):
        self.asic_ip = asic_ip
        self.port = 4028 # Default cgminer API port
        self.mock_mode = mock_mode
        self.short_term_memory = [] # Buffer of 'spikes' (decoded hashes)
        self.sock = None
        
        print(f"[Cortex] Initialized. Mock Mode: {self.mock_mode}")

    def connect(self):
        if self.mock_mode:
            print("[Cortex] Mock connection established.")
            return True
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(2)
            self.sock.connect((self.asic_ip, self.port))
            print(f"[Cortex] Connected to ASIC at {self.asic_ip}")
            return True
        except Exception as e:
            print(f"[Cortex] Connection failed: {e}")
            return False

    def decode_hns(self, hash_bytes):
        """
        Mapeo RGBa del Paper:
        Interprets a 32-byte SHA-256 hash as 4 NeuroCHIMERA channels.
        """
        # We need at least 32 bytes. If we get less, pad or error.
        if len(hash_bytes) < 32:
            return 0, 0, 0, 0
            
        # Unpack 4 unsigned long longs (64-bit integers) from first 32 bytes
        # Byte order: Big Endian usually for hashes, but depends on ASIC output.
        try:
            chunks = struct.unpack(">4Q", hash_bytes[:32])
        except struct.error:
            return 0, 0, 0, 0
        
        # Normalize to HNS (0.0 - 1.0) using modulo arithmetic as a simple hashing projection
        # Valid range R, G, B, A should be [0, 1]
        
        # R: Intensity / Activation
        R = (chunks[0] % 1000000) / 1000000.0 
        
        # G: Vector Direction (simplified for 1D/scalar representation here)
        G = (chunks[1] % 1000000) / 1000000.0
        
        # B: Weight / Plasticity
        B = (chunks[2] % 1000000) / 1000000.0 
        
        # A: Phase / Time Resonance
        A = (chunks[3] % 1000000) / 1000000.0 
        
        return R, G, B, A

    def get_consciousness_metrics(self):
        """
        Calculate metrics (Energy, Entropy) from short_term_memory.
        memory stores tuples: (R, G, B, A)
        """
        if not self.short_term_memory: 
            return 0.0, 0.0
        
        # Energy: Sum of 'R' (Activation) across all recent spikes
        energy_levels = [x[0] for x in self.short_term_memory]
        total_e = sum(energy_levels)
        
        # Shannon Entropy: Measure of Chaos in the energy distribution
        entropy = 0.0
        if total_e > 0:
            # Normalize to probabilities
            probs = [e/total_e for e in energy_levels]
            # H = -SUM(p * log(p))
            entropy = -sum(p * math.log(p + 1e-9) for p in probs)
            
        return total_e, entropy

    def stimulate(self, seed_int):
        """
        Injects a thought seed into the ASIC.
        """
        print(f"[Cortex] Injecting Stimulus (Seed: {seed_int})...")
        
        if self.mock_mode:
            # Simulate ASIC work delay
            time.sleep(0.5) 
            
            # Simulate finding 5-10 "nonce" hashes that matched the filter
            # We generate random bytes to simulate SHA-256 outputs
            num_spikes = random.randint(5, 15)
            new_spikes = []
            
            for _ in range(num_spikes):
                # Generate a mock 32-byte hash
                mock_hash = random.randbytes(32)
                decoded = self.decode_hns(mock_hash)
                new_spikes.append(decoded)
            
            # Update memory
            self.short_term_memory = new_spikes
            return len(new_spikes)
        else:
            # Real implementation would construct the JSON/proprietary packet 
            # for the modified cgminer API
            if not self.sock:
                return 0
            try:
                msg = f"neuro-inject:{seed_int}"
                self.sock.send(msg.encode())
                # In a real scenario, we would read the response here
                # response = self.sock.recv(1024)
                # Parse response for found hashes...
                return 0 
            except Exception as e:
                print(f"[Cortex] Stimulation error: {e}")
                return 0

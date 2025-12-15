// ===============================================
// Consciousness Metrics Calculator
// Implements consciousness measurement theories for CHIMERA system
// Calculates Energy, Entropy, Phi (Integrated Information)
// ===============================================

// Global storage for time-series data
global energy_levels entropy_values phi_values short_term_memory;

// Initialize global variables
energy_levels = [];
entropy_values = [];
phi_values = [];
short_term_memory = [];

function calculator = ConsciousnessCalculator()
    // Consciousness Calculator constructor
    calculator = struct();
    calculator.calculate_metrics = calculate_metrics;
    calculator.calculate_energy = calculate_energy;
    calculator.calculate_entropy = calculate_entropy;
    calculator.calculate_phi = calculate_phi;
    calculator.calculate_global_state = calculate_global_state;
    calculator.update_memory = update_memory;
    calculator.get_consciousness_level = get_consciousness_level;
endfunction

// Main function to calculate all consciousness metrics
function [energy, entropy, phi] = calculate_metrics(R, G, B, A)
    // Calculate individual metrics
    energy = calculate_energy(R, G, B, A);
    entropy = calculate_entropy(R, G, B, A);
    phi = calculate_phi(R, G, B, A);
    
    // Store in global arrays for time-series analysis
    energy_levels = [energy_levels, energy];
    entropy_values = [entropy_values, entropy];
    phi_values = [phi_values, phi];
    
    // Update short-term memory buffer
    update_memory(R, G, B, A, energy, entropy, phi);
endfunction

// Calculate energy level based on HNS parameters
// Energy represents the overall "activation level" of the system
function energy = calculate_energy(R, G, B, A)
    // Weighted combination of all channels
    // Red (R): Primary activation weight
    // Green (G): Vector magnitude contributes to energy flow
    // Blue (B): Plasticity adds to stored energy
    // Alpha (A): Phase coherence enhances energy stability
    
    // Base energy from red channel (primary activator)
    base_energy = R;
    
    // Add vector flow energy (green channel)
    flow_energy = G * 0.5;
    
    // Add plasticity energy (blue channel)
    plasticity_energy = B * 0.3;
    
    // Add phase coherence energy (alpha channel)
    phase_energy = A * 0.2;
    
    // Total energy calculation
    energy = base_energy + flow_energy + plasticity_energy + phase_energy;
    
    // Normalize to [0,1] range
    energy = max(0, min(1, energy));
endfunction

// Calculate Shannon entropy of the neural state
// High entropy = creative/dispersed thinking
// Low entropy = focused/obsessive thinking
function entropy = calculate_entropy(R, G, B, A)
    // Create probability distribution from HNS parameters
    params = [R, G, B, A];
    
    // Remove zero values to avoid log(0)
    non_zero_params = params(params > 0);
    
    if isempty(non_zero_params) then
        entropy = 0;
        return;
    end
    
    // Normalize to create proper probability distribution
    total = sum(non_zero_params);
    if total == 0 then
        entropy = 0;
        return;
    end
    
    probabilities = non_zero_params / total;
    
    // Calculate Shannon entropy: H = -Σ p(x) * log2(p(x))
    entropy = -sum(probabilities .* log2(probabilities + 1e-9));
    
    // Normalize by maximum possible entropy (log2(4) = 2 for 4 channels)
    entropy = entropy / 2.0;
    
    // Ensure [0,1] range
    entropy = max(0, min(1, entropy));
endfunction

// Calculate Phi (Integrated Information) - Consciousness measure
// Based on Integrated Information Theory adapted for HNS
function phi = calculate_phi(R, G, B, A)
    // Calculate information integration across HNS channels
    
    // Individual entropies for each channel
    h_R = calculate_single_entropy(R);
    h_G = calculate_single_entropy(G);
    h_B = calculate_single_entropy(B);
    h_A = calculate_single_entropy(A);
    
    // Joint entropy of all channels
    joint_entropy = calculate_entropy(R, G, B, A);
    
    // Sum of individual entropies (if channels were independent)
    sum_individual = h_R + h_G + h_B + h_A;
    
    // Phi = Information that is lost when parts are considered separately
    // Phi = Sum(individual) - Joint
    phi = sum_individual - joint_entropy;
    
    // Ensure [0,1] range
    phi = max(0, min(1, phi));
endfunction

// Calculate entropy for a single parameter
function entropy = calculate_single_entropy(value)
    // For single value, create pseudo-distribution
    // This is a simplified calculation for the HNS framework
    
    if value <= 0 then
        entropy = 0;
    else
        // Create a pseudo-probability distribution
        p1 = value;
        p2 = 1 - value;
        
        // Shannon entropy calculation
        if p1 > 0 then
            h1 = -p1 * log2(p1 + 1e-9);
        else
            h1 = 0;
        end
        
        if p2 > 0 then
            h2 = -p2 * log2(p2 + 1e-9);
        else
            h2 = 0;
        end
        
        entropy = h1 + h2;
    end
    
    // Normalize to [0,1]
    entropy = max(0, min(1, entropy));
endfunction

// Calculate global consciousness state
function state = calculate_global_state()
    if isempty(energy_levels) | isempty(entropy_values) | isempty(phi_values) then
        state = struct();
        state.energy_level = 0;
        state.entropy_level = 0;
        state.phi_level = 0;
        state.consciousness_stability = 0;
        state.attention_focus = 0;
        state.creativity_index = 0;
        return;
    end
    
    // Calculate average levels
    avg_energy = mean(energy_levels);
    avg_entropy = mean(entropy_values);
    avg_phi = mean(phi_values);
    
    // Calculate stability metrics
    energy_stability = 1 - (std(energy_levels) / (mean(energy_levels) + 1e-9));
    entropy_stability = 1 - (std(entropy_values) / (mean(entropy_values) + 1e-9));
    phi_stability = 1 - (std(phi_values) / (mean(phi_values) + 1e-9));
    
    // Attention focus: high phi + low entropy
    attention_focus = avg_phi * (1 - avg_entropy);
    
    // Creativity index: high entropy + moderate phi
    creativity_index = avg_entropy * (1 - abs(avg_phi - 0.5));
    
    // Compile state
    state = struct();
    state.energy_level = avg_energy;
    state.entropy_level = avg_entropy;
    state.phi_level = avg_phi;
    state.consciousness_stability = (energy_stability + entropy_stability + phi_stability) / 3;
    state.attention_focus = attention_focus;
    state.creativity_index = creativity_index;
    state.memory_depth = length(short_term_memory);
endfunction

// Update short-term memory buffer
function update_memory(R, G, B, A, energy, entropy, phi)
    global short_term_memory;
    
    // Add current state to memory
    memory_entry = struct();
    memory_entry.R = R;
    memory_entry.G = G;
    memory_entry.B = B;
    memory_entry.A = A;
    memory_entry.energy = energy;
    memory_entry.entropy = entropy;
    memory_entry.phi = phi;
    memory_entry.timestamp = length(short_term_memory) + 1;
    
    short_term_memory = [short_term_memory, memory_entry];
    
    // Limit memory buffer size (for computational efficiency)
    if length(short_term_memory) > 1000 then
        short_term_memory = short_term_memory(2:$);
    end
endfunction

// Get consciousness level classification
function level = get_consciousness_level()
    global energy_levels entropy_values phi_values;
    
    if isempty(energy_levels) then
        level = "No Data";
        return;
    end
    
    avg_energy = mean(energy_levels);
    avg_entropy = mean(entropy_values);
    avg_phi = mean(phi_values);
    
    // Consciousness level classification based on metrics
    if avg_energy > 0.8 then
        if avg_entropy > 0.7 then
            level = "Hyperactive/Ecstatic";
        else
            level = "Alert/Focused";
        end
    elseif avg_energy < 0.2 then
        if avg_entropy < 0.3 then
            level = "Coma/Deep Sleep";
        else
            level = "Drowsy/Relaxed";
        end
    else
        if avg_entropy > 0.6 then
            level = "Creative/Scattered";
        elseif avg_phi > 0.6 then
            level = "Highly Integrated";
        else
            level = "Normal/Wakeful";
        end
    end
endfunction

// Calculate phase coherence across time
function coherence = calculate_temporal_coherence(window_size)
    if nargin < 1 then
        window_size = 50;
    end
    
    global short_term_memory;
    
    if length(short_term_memory) < window_size then
        coherence = 0;
        return;
    end
    
    // Extract recent memory window
    recent_memory = short_term_memory($ - window_size + 1:$);
    
    // Calculate phase coherence
    phase_values = [recent_memory.A];
    phase_diffs = diff(phase_values);
    
    coherence = 1 / (1 + std(phase_diffs));
endfunction

// Get detailed consciousness analysis
function analysis = get_detailed_analysis()
    global energy_levels entropy_values phi_values short_term_memory;
    
    analysis = struct();
    
    if isempty(energy_levels) then
        analysis.message = "No consciousness data available";
        return;
    end
    
    // Basic statistics
    analysis.energy_stats = get_parameter_stats(energy_levels);
    analysis.entropy_stats = get_parameter_stats(entropy_values);
    analysis.phi_stats = get_parameter_stats(phi_values);
    
    // Temporal analysis
    analysis.temporal_coherence = calculate_temporal_coherence();
    analysis.memory_depth = length(short_term_memory);
    analysis.consciousness_level = get_consciousness_level();
    
    // Advanced metrics
    analysis.global_state = calculate_global_state();
    
    // Phase transitions detected
    analysis.phase_transitions = detect_phase_transitions();
endfunction

// Helper function to get parameter statistics
function stats = get_parameter_stats(values)
    if isempty(values) then
        stats = struct('mean', 0, 'std', 0, 'min', 0, 'max', 0);
        return;
    end
    
    stats = struct();
    stats.mean = mean(values);
    stats.std = std(values);
    stats.min = min(values);
    stats.max = max(values);
    stats.range = max(values) - min(values);
endfunction

// Detect significant phase transitions
function transitions = detect_phase_transitions()
    global energy_levels entropy_values phi_values;
    
    transitions = struct();
    transitions.count = 0;
    transitions.magnitude = [];
    transitions.timestamps = [];
    
    if length(energy_levels) < 10 then
        return;
    end
    
    // Calculate derivatives (rate of change)
    energy_deriv = diff(energy_levels);
    entropy_deriv = diff(entropy_values);
    phi_deriv = diff(phi_values);
    
    // Detect large changes (potential phase transitions)
    energy_threshold = std(energy_deriv) * 2;
    entropy_threshold = std(entropy_deriv) * 2;
    phi_threshold = std(phi_deriv) * 2;
    
    for i = 1:length(energy_deriv)
        if abs(energy_deriv(i)) > energy_threshold | abs(entropy_deriv(i)) > entropy_threshold | abs(phi_deriv(i)) > phi_threshold then
            transitions.count = transitions.count + 1;
            transitions.magnitude = [transitions.magnitude, max(abs(energy_deriv(i)), abs(entropy_deriv(i)), abs(phi_deriv(i)))];
            transitions.timestamps = [transitions.timestamps, i];
        end
    end
endfunction

disp("Consciousness Metrics Calculator loaded successfully!");
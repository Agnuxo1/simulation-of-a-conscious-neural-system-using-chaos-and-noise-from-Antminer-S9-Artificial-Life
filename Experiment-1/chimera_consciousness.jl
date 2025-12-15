# ===============================================
# CHIMERA Consciousness Engine in Julia
# Ported from Python/Scilab implementations
# Implements VESELOV consciousness calculations and HNS processing
# Enhanced with ODE dynamics and parallel computation
# ===============================================

using LinearAlgebra
using Statistics
using Random
using DifferentialEquations
using ThreadPools
using Printf

# VESELOV Consciousness System Structure
mutable struct VESELOVConsciousness
    hns_nodes::Int64
    temperature::Float64
    energy_landscape::Array{Float64,3}
    phase_space::Array{ComplexF64,4}
    history::Vector{Dict{String,Float64}}
    # Additional fields for ODE dynamics
    consciousness_state::Vector{Float64}
    time_derivative::Vector{Float64}
    simulation_time::Float64
end

# ===============================================
# HNS Processing Functions
# ===============================================

"""
Normalize RGBA values from hash bytes to [0,1] range
"""
function normalize_rgba(hash_bytes::Vector{UInt8})::Tuple{Float64,Float64,Float64,Float64}
    if length(hash_bytes) != 32
        error("Invalid hash length. Expected 32 bytes, got $(length(hash_bytes))")
    end

    # Extract 8-byte chunks and convert to UInt64
    chunks = zeros(UInt64, 4)
    for i in 1:4
        start_idx = (i-1)*8 + 1
        end_idx = i*8
        chunk_bytes = hash_bytes[start_idx:end_idx]
        chunks[i] = reinterpret(UInt64, reverse(chunk_bytes))[1]
    end

    # Normalize to [0,1] range using modulo operation
    normalization_factor = 1000000.0
    R = (chunks[1] % Int(normalization_factor)) / normalization_factor
    G = (chunks[2] % Int(normalization_factor)) / normalization_factor
    B = (chunks[3] % Int(normalization_factor)) / normalization_factor
    A = (chunks[4] % Int(normalization_factor)) / normalization_factor

    # Ensure valid ranges
    R = clamp(R, 0.0, 1.0)
    G = clamp(G, 0.0, 1.0)
    B = clamp(B, 0.0, 1.0)
    A = clamp(A, 0.0, 1.0)

    return R, G, B, A
end

"""
Calculate vector magnitude from green channel data (3D torus representation)
"""
function calculate_vector_magnitude(green_value::UInt64)::Float64
    # Extract 3D vector components from the 64-bit value
    dx = ((green_value >> 40) & 0xFF) - 128
    dy = ((green_value >> 24) & 0xFF) - 128
    dz = ((green_value >> 8) & 0xFF) - 128

    # Normalize to [-1, 1] range and calculate magnitude
    dx_norm = dx / 128.0
    dy_norm = dy / 128.0
    dz_norm = dz / 128.0

    magnitude = sqrt(dx_norm^2 + dy_norm^2 + dz_norm^2) / sqrt(3.0)
    return clamp(magnitude, 0.0, 1.0)
end

"""
Calculate phase coherence from temporal data
"""
function calculate_phase_coherence(phase_data::Vector{Float64})::Float64
    if length(phase_data) < 2
        return 0.0
    end

    # Calculate phase differences
    phase_diff = diff(phase_data)

    # Coherence measure: inverse of variance in phase differences
    coherence = 1.0 / (1.0 + var(phase_diff))
    return clamp(coherence, 0.0, 1.0)
end

# ===============================================
# Consciousness Metrics Functions
# ===============================================

"""
Calculate energy level based on weighted RGBA combination
Energy represents overall activation level of the system
"""
function calculate_energy(R::Float64, G::Float64, B::Float64, A::Float64)::Float64
    # Weighted combination of all channels
    # R: Primary activation (40%), G: Vector flow (30%), B: Plasticity (20%), A: Phase coherence (10%)
    energy = R * 0.4 + G * 0.3 + B * 0.2 + A * 0.1

    # Normalize to [0,1] range
    return clamp(energy, 0.0, 1.0)
end

"""
Calculate Shannon entropy of the neural state
High entropy = creative/dispersed thinking
Low entropy = focused/obsessive thinking
"""
function compute_entropy(R::Float64, G::Float64, B::Float64, A::Float64)::Float64
    params = [R, G, B, A]

    # Remove zero values to avoid log(0)
    non_zero_params = filter(x -> x > 0, params)

    if isempty(non_zero_params)
        return 0.0
    end

    # Normalize to create proper probability distribution
    total = sum(non_zero_params)
    probabilities = non_zero_params ./ total

    # Calculate Shannon entropy: H = -Σ p(x) * log2(p(x))
    entropy = -sum(probabilities .* log2.(probabilities .+ 1e-9))

    # Normalize by maximum possible entropy (log2(4) = 2 for 4 channels)
    entropy = entropy / 2.0

    return clamp(entropy, 0.0, 1.0)
end

"""
Calculate Phi (Integrated Information) - Consciousness measure
Based on Integrated Information Theory adapted for HNS
"""
function compute_phi(R::Float64, G::Float64, B::Float64, A::Float64)::Float64
    # Individual entropies for each channel
    h_R = calculate_single_entropy(R)
    h_G = calculate_single_entropy(G)
    h_B = calculate_single_entropy(B)
    h_A = calculate_single_entropy(A)

    # Joint entropy of all channels
    joint_entropy = compute_entropy(R, G, B, A)

    # Sum of individual entropies (if channels were independent)
    sum_individual = h_R + h_G + h_B + h_A

    # Phi = Information that is lost when parts are considered separately
    # Phi = Sum(individual) - Joint
    phi = sum_individual - joint_entropy

    return clamp(phi, 0.0, 1.0)
end

"""
Calculate entropy for a single parameter
"""
function calculate_single_entropy(value::Float64)::Float64
    if value <= 0.0
        return 0.0
    end

    # Create pseudo-probability distribution
    p1 = value
    p2 = 1.0 - value

    # Shannon entropy calculation
    h1 = p1 > 0 ? -p1 * log2(p1 + 1e-9) : 0.0
    h2 = p2 > 0 ? -p2 * log2(p2 + 1e-9) : 0.0

    entropy = h1 + h2
    return clamp(entropy, 0.0, 1.0)
end

# ===============================================
# Core Consciousness Functions
# ===============================================

"""
Apply temperature modulation to consciousness parameters
"""
function temperature_modulation(value::Float64, temperature::Float64)::Float64
    # Temperature effects on consciousness processing
    # Higher temperature increases noise/randomness
    # Lower temperature increases stability/focus

    if temperature > 1.0
        # High temperature: add noise
        noise = randn() * 0.1 * (temperature - 1.0)
        modulated = value + noise
    elseif temperature < 1.0
        # Low temperature: increase stability
        stability_factor = 1.0 - (1.0 - temperature) * 0.5
        modulated = value * stability_factor + (1.0 - stability_factor) * 0.5
    else
        modulated = value
    end

    return clamp(modulated, 0.0, 1.0)
end

"""
Detect phase transitions in consciousness state
"""
function detect_phase_transition(system::VESELOVConsciousness, i::Int, j::Int, k::Int)::Float64
    if length(system.history) < 5
        return 0.0
    end

    # Get recent history for this node
    recent_history = system.history[max(1, end-9):end]

    # Extract energy values for trend analysis
    energies = [h["energy"] for h in recent_history]

    if length(energies) < 3
        return 0.0
    end

    # Calculate rate of change
    derivatives = diff(energies)
    recent_derivatives = derivatives[max(1, end-4):end]

    # Detect significant changes (potential phase transitions)
    mean_deriv = mean(abs.(recent_derivatives))
    std_deriv = std(abs.(recent_derivatives))

    if std_deriv > 0
        transition_strength = mean_deriv / std_deriv
        return clamp(transition_strength, 0.0, 1.0)
    end

    return 0.0
end

# ===============================================
# Attention Mechanisms
# ===============================================

"""
Implement query-key-value attention mechanism for consciousness processing
"""
function attention_mechanism(query::Vector{Float64}, keys::Matrix{Float64}, values::Matrix{Float64})::Vector{Float64}
    # Simplified attention mechanism for consciousness processing

    n_keys = size(keys, 2)
    attention_weights = zeros(n_keys)

    # Calculate attention scores (dot product similarity)
    for i in 1:n_keys
        key = keys[:, i]
        score = dot(query, key) / (norm(query) * norm(key) + 1e-9)
        attention_weights[i] = score
    end

    # Apply softmax to get attention weights
    max_score = maximum(attention_weights)
    exp_weights = exp.(attention_weights .- max_score)
    attention_weights = exp_weights ./ sum(exp_weights)

    # Apply attention to values
    attended_output = zeros(size(values, 1))
    for i in 1:n_keys
        attended_output .+= attention_weights[i] .* values[:, i]
    end

    return attended_output
end

"""
Multi-head attention for complex consciousness processing
"""
function multihead_attention(query::Vector{Float64}, keys::Matrix{Float64}, values::Matrix{Float64},
                           num_heads::Int=4)::Vector{Float64}
    d_model = length(query)
    head_dim = d_model ÷ num_heads

    attention_outputs = []

    for h in 1:num_heads
        start_idx = (h-1) * head_dim + 1
        end_idx = h * head_dim

        q_head = query[start_idx:end_idx]
        k_head = keys[start_idx:end_idx, :]
        v_head = values[start_idx:end_idx, :]

        head_output = attention_mechanism(q_head, k_head, v_head)
        push!(attention_outputs, head_output)
    end

    # Concatenate heads
    concatenated = vcat(attention_outputs...)

    # Simple feed-forward (identity for now)
    return concatenated
end

# ===============================================
# Phase Transition Analysis
# ===============================================

"""
Calculate order parameter for phase transition detection
"""
function calculate_order_parameter(system::VESELOVConsciousness)::Float64
    if length(system.history) < 5
        return 0.0
    end

    recent_history = system.history[max(1, end-19):end]

    # Extract RGBA values
    R_values = [h["R"] for h in recent_history]
    G_values = [h["G"] for h in recent_history]
    B_values = [h["B"] for h in recent_history]
    A_values = [h["A"] for h in recent_history]

    # Calculate synchronization measures
    rg_corr = cor(R_values, G_values)
    rb_corr = cor(R_values, B_values)
    ga_corr = cor(G_values, A_values)

    # Order parameter as average absolute correlation
    order_param = mean(abs.([rg_corr, rb_corr, ga_corr]))

    return isnan(order_param) ? 0.0 : clamp(order_param, 0.0, 1.0)
end

"""
Detect critical points in consciousness evolution
"""
function detect_critical_points(system::VESELOVConsciousness)::Vector{Int}
    if length(system.history) < 10
        return Int[]
    end

    energies = [h["energy"] for h in system.history]

    # Calculate first and second derivatives
    first_deriv = diff(energies)
    second_deriv = diff(first_deriv)

    critical_points = Int[]

    for i in 5:length(first_deriv)-4
        # Look for significant changes in derivative
        local_window = first_deriv[i-4:i+4]
        local_var = var(local_window)
        global_var = var(first_deriv)

        if local_var > global_var * 2.0
            push!(critical_points, i+1)
        end
    end

    return critical_points
end

"""
Estimate correlation dimension for phase space analysis
"""
function correlation_dimension(system::VESELOVConsciousness, embedding_dim::Int=3)::Float64
    if length(system.history) < embedding_dim * 2 + 10
        return 0.0
    end

    # Use energy values for reconstruction
    data = [h["energy"] for h in system.history]

    # Time delay embedding
    time_delay = 1
    n_points = length(data)
    n_embedded = n_points - (embedding_dim - 1) * time_delay

    if n_embedded < 20
        return 0.0
    end

    # Create embedded trajectory
    trajectory = zeros(n_embedded, embedding_dim)
    for i in 1:n_embedded
        for j in 1:embedding_dim
            idx = i + (j-1) * time_delay
            trajectory[i, j] = idx <= n_points ? data[idx] : 0.0
        end
    end

    # Calculate correlation sum for different radii
    r_values = range(0.01, 0.5, length=20)
    correlation_sums = zeros(length(r_values))

    for (r_idx, r) in enumerate(r_values)
        count = 0
        for i in 1:n_embedded
            for j in i+1:n_embedded
                if norm(trajectory[i, :] - trajectory[j, :]) < r
                    count += 1
                end
            end
        end
        correlation_sums[r_idx] = 2 * count / (n_embedded * (n_embedded - 1))
    end

    # Estimate dimension from slope of log-log plot
    valid_indices = findall(x -> 0 < x < 1, correlation_sums)
    if length(valid_indices) > 5
        log_r = log.(r_values[valid_indices])
        log_c = log.(correlation_sums[valid_indices])

        # Linear regression to find slope
        x_mean = mean(log_r)
        y_mean = mean(log_c)
        slope = sum((log_r .- x_mean) .* (log_c .- y_mean)) / sum((log_r .- x_mean).^2)

        return clamp(slope, 0.0, embedding_dim * 1.0)
    end

    return embedding_dim * 1.0
end

# ===============================================
# ODE-Based Consciousness Dynamics
# ===============================================

"""
Compute consciousness derivatives for ODE system
This function defines the differential equations for consciousness evolution
"""
function compute_consciousness_derivative!(du::Vector{Float64}, u::Vector{Float64}, 
                                        system::VESELOVConsciousness, t::Float64)
    # Unpack consciousness state
    n_nodes = system.hns_nodes
    total_elements = n_nodes^3 * 4  # 4 consciousness dimensions per node

    if length(u) != total_elements
        error("State vector length mismatch: expected $total_elements, got $(length(u))")
    end

    # Reshape state into manageable format
    consciousness_3d = reshape(u, n_nodes, n_nodes, n_nodes, 4)

    # Initialize derivatives
    du .= 0.0

    # Parallel computation for large systems
    @threads for i in 1:n_nodes
        for j in 1:n_nodes
            for k in 1:n_nodes
                idx_3d = i + (j-1)*n_nodes + (k-1)*n_nodes^2
                
                # Get local consciousness values
                entropy_val = consciousness_3d[i, j, k, 1]
                phi_val = consciousness_3d[i, j, k, 2]
                energy_val = consciousness_3d[i, j, k, 3]
                phase_val = consciousness_3d[i, j, k, 4]

                # Get local energy landscape
                local_energy = system.energy_landscape[i, j, k]

                # VESELOV consciousness dynamics equations
                # Entropy evolution: du/dt = f(energy, temperature, coupling)
                d_entropy = -entropy_val * local_energy + 
                           system.temperature * (0.5 - entropy_val) +
                           0.1 * randn()

                # Phi evolution: information integration dynamics
                d_phi = phi_val * (1.0 - phi_val) * (entropy_val - 0.5) +
                       0.05 * randn()

                # Energy dynamics: coupling with landscape
                d_energy = -0.5 * energy_val + 
                          0.3 * local_energy +
                          0.2 * sin(t * 0.1) +
                          0.05 * randn()

                # Phase transition dynamics
                d_phase = phase_val * (1.0 - phase_val) * 
                         (energy_val - phi_val) +
                         0.03 * randn()

                # Update derivatives with neighbor coupling
                # Add spatial coupling for emergent consciousness
                if i > 1
                    d_entropy += 0.1 * (consciousness_3d[i-1, j, k, 1] - entropy_val)
                    d_phi += 0.1 * (consciousness_3d[i-1, j, k, 2] - phi_val)
                    d_energy += 0.1 * (consciousness_3d[i-1, j, k, 3] - energy_val)
                    d_phase += 0.1 * (consciousness_3d[i-1, j, k, 4] - phase_val)
                end

                if i < n_nodes
                    d_entropy += 0.1 * (consciousness_3d[i+1, j, k, 1] - entropy_val)
                    d_phi += 0.1 * (consciousness_3d[i+1, j, k, 2] - phi_val)
                    d_energy += 0.1 * (consciousness_3d[i+1, j, k, 3] - energy_val)
                    d_phase += 0.1 * (consciousness_3d[i+1, j, k, 4] - phase_val)
                end

                # Write back derivatives
                du[idx_3d] = d_entropy
                du[idx_3d + n_nodes^3] = d_phi
                du[idx_3d + 2*n_nodes^3] = d_energy
                du[idx_3d + 3*n_nodes^3] = d_phase
            end
        end
    end
end

# ===============================================
# System Initialization and Processing
# ===============================================

"""
Initialize VESELOV consciousness system
"""
function initialize_consciousness_system(nodes::Int, temperature::Float64=1.0)::VESELOVConsciousness
    # Initialize 3D HNS grid
    energy_landscape = randn(nodes, nodes, nodes) * 0.1

    # Create phase space for complex consciousness dynamics (RGBA dimensions)
    phase_space = zeros(ComplexF64, nodes, nodes, nodes, 4)

    # Initialize consciousness state vector for ODE system
    total_elements = nodes^3 * 4
    consciousness_state = rand(total_elements) * 0.5 + 0.25  # Initial values in [0.25, 0.75]
    time_derivative = zeros(total_elements)

    # Initialize history
    history = Vector{Dict{String,Float64}}()

    return VESELOVConsciousness(nodes, temperature, energy_landscape, phase_space, 
                               history, consciousness_state, time_derivative, 0.0)
end

"""
Compute consciousness metrics for a single node
"""
function compute_consciousness_metrics!(system::VESELOVConsciousness, i::Int, j::Int, k::Int)
    # Get current energy
    local_energy = system.energy_landscape[i, j, k]

    # (simplified - Generate RGBA values in real implementation from hash)
    R = clamp(local_energy + randn() * 0.1, 0.0, 1.0)
    G = clamp(abs(local_energy) + randn() * 0.05, 0.0, 1.0)
    B = clamp(local_energy * 0.5 + 0.5 + randn() * 0.08, 0.0, 1.0)
    A = clamp(sin(local_energy * π) * 0.5 + 0.5 + randn() * 0.03, 0.0, 1.0)

    # Apply temperature modulation
    R = temperature_modulation(R, system.temperature)
    G = temperature_modulation(G, system.temperature)
    B = temperature_modulation(B, system.temperature)
    A = temperature_modulation(A, system.temperature)

    # Calculate consciousness metrics
    energy = calculate_energy(R, G, B, A)
    entropy = compute_entropy(R, G, B, A)
    phi = compute_phi(R, G, B, A)

    # Store in phase space
    system.phase_space[i, j, k, 1] = complex(entropy, 0.0)
    system.phase_space[i, j, k, 2] = complex(phi, 0.0)
    system.phase_space[i, j, k, 3] = complex(energy, 0.0)
    system.phase_space[i, j, k, 4] = complex(detect_phase_transition(system, i, j, k), 0.0)

    # Update history
    push!(system.history, Dict(
        "R" => R, "G" => G, "B" => B, "A" => A,
        "energy" => energy, "entropy" => entropy, "phi" => phi
    ))

    # Limit history size
    if length(system.history) > 1000
        system.history = system.history[end-999:end]
    end
end

"""
Run consciousness simulation for the entire system (legacy version for compatibility)
"""
function run_consciousness_simulation(system::VESELOVConsciousness)
    nodes = system.hns_nodes

    # Process all nodes
    for i in 1:nodes
        for j in 1:nodes
            for k in 1:nodes
                compute_consciousness_metrics!(system, i, j, k)
            end
        end
    end
end

"""
Enhanced consciousness simulation with ODE dynamics and parallel computation
"""
function run_consciousness_simulation(system::VESELOVConsciousness; 
                                    duration::Float64=10.0, 
                                    save_interval::Float64=0.1,
                                    use_ode::Bool=true)
    
    nodes = system.hns_nodes
    total_elements = nodes^3 * 4

    if use_ode
        println("Running ODE-based consciousness simulation with $nodes^3 nodes...")
        println("Total state dimensions: $total_elements")
        
        # Create ODE problem for consciousness evolution
        function consciousness_ode!(du, u, p, t)
            compute_consciousness_derivative!(du, u, p, t)
        end

        # Initial conditions from current state or random if empty
        if isempty(system.history)
            u0 = rand(total_elements) * 0.5 + 0.25
        else
            # Use last known state from history
            last_state = system.history[end]
            u0 = [last_state["entropy"], last_state["phi"], last_state["energy"], 
                  last_state["R"], last_state["G"], last_state["B"], last_state["A"]]
            # Replicate for all nodes (simplified)
            u0 = repeat(u0, total_elements ÷ 7)
        end

        # Create and solve ODE problem
        prob = ODEProblem(consciousness_ode!, u0, (0.0, duration), system)
        
        # Use Tsit5 adaptive solver for efficient time stepping
        @time sol = solve(prob, Tsit5(), saveat=save_interval, 
                         reltol=1e-6, abstol=1e-8, maxiters=10000)
        
        # Update system state with solution
        system.consciousness_state = sol.u[end]
        system.simulation_time = duration
        
        # Extract global metrics from final state
        final_state = sol.u[end]
        global_entropy = mean(final_state[1:total_elements÷4])
        global_phi = mean(final_state[total_elements÷4+1:total_elements÷2])
        global_energy = mean(final_state[total_elements÷2+1:3*total_elements÷4])
        
        # Update history with ODE results
        push!(system.history, Dict(
            "time" => duration,
            "entropy" => global_entropy,
            "phi" => global_phi,
            "energy" => global_energy,
            "R" => global_energy, "G" => global_entropy, "B" => global_phi, "A" => 0.5
        ))
        
        println("ODE simulation completed. Final state: entropy=$(round(global_entropy, digits=4)), phi=$(round(global_phi, digits=4)), energy=$(round(global_energy, digits=4))")
        
        return sol
    else
        # Legacy sequential simulation
        println("Running legacy consciousness simulation with $nodes^3 nodes...")
        @time run_consciousness_simulation(system)
        return nothing
    end
end

# ===============================================
# Performance Benchmarking
# ===============================================

"""
Benchmark consciousness simulation performance for different node counts
"""
function benchmark_consciousness_performance(;node_counts::Vector{Int}=[10, 20, 50, 100],
                                           temperature::Float64=1.0,
                                           duration::Float64=5.0)
    println("=== Consciousness Engine Performance Benchmarks ===")
    println("Temperature: $temperature, Duration: $duration")
    println("="^60)
    
    results = Dict{Int, Dict{String, Float64}}()
    
    for nodes in node_counts
        println("\n--- Benchmarking $nodes^3 = $(nodes^3) nodes ---")
        
        # Test legacy simulation
        println("Legacy simulation:")
        system_legacy = initialize_consciousness_system(nodes, temperature)
        @time run_consciousness_simulation(system_legacy)
        
        # Test ODE simulation
        println("ODE simulation:")
        system_ode = initialize_consciousness_system(nodes, temperature)
        @time sol = run_consciousness_simulation(system_ode, duration=duration, use_ode=true)
        
        # Store results
        results[nodes] = Dict(
            "nodes" => nodes^3,
            "legacy_time" => @elapsed run_consciousness_simulation(system_legacy),
            "ode_time" => @elapsed run_consciousness_simulation(system_ode, duration=duration, use_ode=true),
            "final_entropy" => system_ode.history[end]["entropy"],
            "final_phi" => system_ode.history[end]["phi"],
            "final_energy" => system_ode.history[end]["energy"]
        )
        
        println("Results: nodes=$(nodes^3), legacy_time=$(round(results[nodes]["legacy_time"], digits=3))s, " *
                "ode_time=$(round(results[nodes]["ode_time"], digits=3))s")
    end
    
    # Print summary
    println("\n=== Performance Summary ===")
    println("Nodes\t\tLegacy(s)\tODE(s)\t\tSpeedup\t\tEfficiency")
    println("-"^70)
    
    for nodes in node_counts
        result = results[nodes]
        speedup = result["legacy_time"] / result["ode_time"]
        efficiency = speedup / Threads.nthreads()
        @printf "%d\t\t%.3f\t\t%.3f\t\t%.2fx\t\t%.2f\n" 
                result["nodes"] result["legacy_time"] result["ode_time"] speedup efficiency
    end
    
    return results
end

# ===============================================
# Global Consciousness State
# ===============================================

"""
Get global consciousness state
"""
function get_global_consciousness_state(system::VESELOVConsciousness)::Dict{String,Float64}
    if isempty(system.history)
        return Dict(
            "energy_level" => 0.5,
            "entropy_level" => 0.5,
            "phi_level" => 0.5,
            "order_parameter" => 0.0,
            "correlation_dimension" => 0.0
        )
    end

    recent_history = system.history[max(1, end-49):end]  # Last 50 states

    energies = [h["energy"] for h in recent_history]
    entropies = [h["entropy"] for h in recent_history]
    phis = [h["phi"] for h in recent_history]

    return Dict(
        "energy_level" => mean(energies),
        "entropy_level" => mean(entropies),
        "phi_level" => mean(phis),
        "order_parameter" => calculate_order_parameter(system),
        "correlation_dimension" => correlation_dimension(system)
    )
end

# ===============================================
# Sample Data Testing
# ===============================================

"""
Generate sample hash data for testing
"""
function generate_sample_hash(seed::Int=42)::Vector{UInt8}
    Random.seed!(seed)
    return rand(UInt8, 32)
end

"""
Test the consciousness system with sample data
"""
function test_consciousness_system()
    println("=== Testing Enhanced CHIMERA Consciousness System ===")

    # Initialize system
    system = initialize_consciousness_system(10, 1.0)
    println("Initialized system with $(system.hns_nodes)^3 nodes")

    # Test legacy simulation
    println("\n--- Testing Legacy Simulation ---")
    run_consciousness_simulation(system)

    state = get_global_consciousness_state(system)
    println("Legacy Results:")
    println("  Energy Level: $(round(state["energy_level"], digits=3))")
    println("  Entropy Level: $(round(state["entropy_level"], digits=3))")
    println("  Phi Level: $(round(state["phi_level"], digits=3))")

    # Test ODE simulation
    println("\n--- Testing ODE Simulation ---")
    sol = run_consciousness_simulation(system, duration=2.0, use_ode=true)
    
    state_ode = get_global_consciousness_state(system)
    println("ODE Results:")
    println("  Energy Level: $(round(state_ode["energy_level"], digits=3))")
    println("  Entropy Level: $(round(state_ode["entropy_level"], digits=3))")
    println("  Phi Level: $(round(state_ode["phi_level"], digits=3))")

    # Test HNS processing
    println("\n--- Testing HNS Processing ---")
    sample_hash = generate_sample_hash()
    R, G, B, A = normalize_rgba(sample_hash)
    println("Sample RGBA: R=$(round(R, digits=3)), G=$(round(G, digits=3)), B=$(round(B, digits=3)), A=$(round(A, digits=3))")

    energy = calculate_energy(R, G, B, A)
    entropy = compute_entropy(R, G, B, A)
    phi = compute_phi(R, G, B, A)
    println("Metrics: Energy=$(round(energy, digits=3)), Entropy=$(round(entropy, digits=3)), Phi=$(round(phi, digits=3))")

    # Test attention mechanism
    println("\n--- Testing Attention Mechanism ---")
    query = [0.5, 0.3, 0.8, 0.1]
    keys = rand(4, 5)
    values = rand(4, 5)
    attended = attention_mechanism(query, keys, values)
    println("Attention output dimension: $(length(attended))")

    # Test performance benchmark
    println("\n--- Performance Benchmark ---")
    benchmark_consciousness_performance(node_counts=[5, 10], duration=1.0)

    println("\n=== Enhanced test completed successfully! ===")
end

# Export main functions
export VESELOVConsciousness, initialize_consciousness_system, run_consciousness_simulation,
       get_global_consciousness_state, test_consciousness_system, normalize_rgba,
       calculate_energy, compute_entropy, compute_phi, attention_mechanism,
       compute_consciousness_derivative!, benchmark_consciousness_performance
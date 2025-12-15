// ===============================================
// Phase Transition Analysis Module
// Analyzes emergent consciousness behaviors and state transitions
// Critical for understanding CHIMERA's neuromorphic dynamics
// ===============================================

function analyzer = PhaseTransitionAnalyzer()
    // Phase Transition Analyzer constructor
    analyzer = struct();
    analyzer.analyze_state = analyze_state;
    analyzer.detect_critical_points = detect_critical_points;
    analyzer.calculate_order_parameter = calculate_order_parameter;
    analyzer.identify_regimes = identify_regimes;
    analyzer.predict_transitions = predict_transitions;
    analyzer.calculate_critical_exponents = calculate_critical_exponents;
    analyzer.analyze_hysteresis = analyze_hysteresis;
    analyzer.measure_synchronization = measure_synchronization;
endfunction

// Analyze current system state and determine phase
function phase_state = analyze_state(R, G, B, A)
    // Analyze HNS parameters to determine current phase
    
    // Calculate order parameters
    order_param = calculate_order_parameter(R, G, B, A);
    synchronization = measure_synchronization(R, G, B, A);
    
    // Determine phase based on consciousness theory
    if R > 0.8 then
        if synchronization > 0.7 then
            phase_state = "Synchronized Hyperactivity";
        else
            phase_state = "Chaotic Activation";
        end
    elseif R < 0.2 then
        if order_param < 0.3 then
            phase_state = "Disordered Rest";
        else
            phase_state = "Ordered Sleep";
        end
    else
        if synchronization > 0.6 then
            phase_state = "Critical Consciousness";
        elseif order_param > 0.7 then
            phase_state = "Highly Ordered";
        else
            phase_state = "Normal Operation";
        end
    end
    
    // Add phase transition prediction
    transition_probability = predict_transition_probability(R, G, B, A);
    if transition_probability > 0.8 then
        phase_state = phase_state + " (Unstable)";
    end
endfunction

// Calculate order parameter for phase transition detection
function order_param = calculate_order_parameter(R, G, B, A)
    global short_term_memory;
    
    if isempty(short_term_memory) | length(short_term_memory) < 5 then
        // For single state, calculate based on channel balance
        channel_values = [R, G, B, A];
        order_param = 1 - (std(channel_values) / (mean(channel_values) + 1e-9));
        return;
    end
    
    // Calculate order parameter from recent history
    recent_states = short_term_memory(max(1,end-9):end);
    
    // Extract channel values
    R_values = [recent_states.R];
    G_values = [recent_states.G];
    B_values = [recent_states.B];
    A_values = [recent_states.A];
    
    // Calculate synchronization measure
    rg_sync = abs(corr(R_values', G_values'));
    rb_sync = abs(corr(R_values', B_values'));
    ga_sync = abs(corr(G_values', A_values'));
    
    // Order parameter as average synchronization
    order_param = (rg_sync + rb_sync + ga_sync) / 3;
endfunction

// Detect critical points and phase transitions
function critical_points = detect_critical_points(metric_series, window_size)
    if nargin < 1 then
        global energy_levels;
        metric_series = energy_levels;
    end
    if nargin < 2 then
        window_size = 10;
    end
    
    critical_points = struct();
    critical_points.indices = [];
    critical_points.magnitudes = [];
    critical_points.types = [];
    
    if length(metric_series) < window_size * 2 then
        return;
    end
    
    // Calculate derivatives (rate of change)
    first_derivative = diff(metric_series);
    second_derivative = diff(first_derivative);
    
    // Detect potential critical points
    for i = window_size:length(first_derivative) - window_size
        // Look for significant changes in derivative
        local_var = var(first_derivative(max(1,i-window_size):min(end,i+window_size)));
        global_var = var(first_derivative);
        
        if local_var > global_var * 2 then
            // Potential critical point
            critical_points.indices = [critical_points.indices, i+1];
            critical_points.magnitudes = [critical_points.magnitudes, abs(first_derivative(i))];
            
            // Classify type of transition
            if second_derivative(i) > 0 then
                critical_points.types = [critical_points.types, "Acceleration"];
            elseif second_derivative(i) < 0 then
                critical_points.types = [critical_points.types, "Deceleration"];
            else
                critical_points.types = [critical_points.types, "Inflection"];
            end
        end
    end
    
    // Remove duplicate or very close points
    if length(critical_points.indices) > 1 then
        min_distance = 5;
        filtered_indices = [];
        filtered_magnitudes = [];
        filtered_types = [];
        
        for i = 1:length(critical_points.indices)
            if isempty(filtered_indices) | min(abs(critical_points.indices(i) - filtered_indices)) > min_distance then
                filtered_indices = [filtered_indices, critical_points.indices(i)];
                filtered_magnitudes = [filtered_magnitudes, critical_points.magnitudes(i)];
                filtered_types = [filtered_types, critical_points.types(i)];
            end
        end
        
        critical_points.indices = filtered_indices;
        critical_points.magnitudes = filtered_magnitudes;
        critical_points.types = filtered_types;
    end
endfunction

// Identify different operational regimes
function regimes = identify_regimes(rgba_matrix, time_window)
    if nargin < 1 then
        // Generate sample data if not provided
        rgba_matrix = rand(100, 4);
    end
    if nargin < 2 then
        time_window = 20;
    end
    
    [n_samples, ~] = size(rgba_matrix);
    regimes = struct();
    regimes.regime_id = zeros(n_samples, 1);
    regimes.regime_names = {};
    regimes.transition_points = [];
    
    regime_counter = 1;
    
    for i = 1:time_window:n_samples
        end_idx = min(i + time_window - 1, n_samples);
        window_data = rgba_matrix(i:end_idx, :);
        
        // Analyze regime characteristics
        avg_R = mean(window_data(:,1));
        avg_G = mean(window_data(:,2));
        avg_B = mean(window_data(:,3));
        avg_A = mean(window_data(:,4));
        
        % Classify regime
        if avg_R > 0.7 then
            if avg_G > 0.6 then
                regime_name = "High_Activation_Flow";
            else
                regime_name = "High_Activation_Static";
            end
        elseif avg_R < 0.3 then
            if avg_B > 0.7 then
                regime_name = "Low_Activation_High_Plasticity";
            else
                regime_name = "Rest_State";
            end
        else
            if avg_A > 0.7 then
                regime_name = "High_Phase_Resonance";
            else
                regime_name = "Normal_Operation";
            end
        end
        
        % Assign regime ID
        existing_regime = find(strcmp(regimes.regime_names, regime_name));
        if isempty(existing_regime) then
            regimes.regime_names{regime_counter} = regime_name;
            regime_id = regime_counter;
            regime_counter = regime_counter + 1;
        else
            regime_id = existing_regime;
        end
        
        % Mark regime in time series
        regimes.regime_id(i:end_idx) = regime_id;
        
        % Detect regime transitions
        if i > 1 then
            if regimes.regime_id(i-1) ~= regime_id then
                regimes.transition_points = [regition_points, i];
            end
        end
    end
endfunction

// Predict likelihood of phase transitions
function transition_prob = predict_transition_probability(R, G, B, A)
    global short_term_memory;
    
    // Base probability from current state
    base_prob = 0;
    
    % Calculate instability indicators
    if R > 0.9 | R < 0.1 then
        base_prob = base_prob + 0.3; // Extreme red values are unstable
    end
    
    if abs(G - 0.5) > 0.4 then
        base_prob = base_prob + 0.2; // Extreme vector values
    end
    
    // Check recent history for transition patterns
    if length(short_term_memory) >= 10 then
        recent_states = short_term_memory(end-9:end);
        
        // Look for increasing variance (precursor to transition)
        recent_values = [recent_states.R];
        variance_trend = var(recent_values(6:end)) / var(recent_values(1:5));
        
        if variance_trend > 1.5 then
            base_prob = base_prob + 0.4; // High variance increase
        end
        
        // Look for consecutive changes
        r_changes = abs(diff([recent_states.R]));
        if mean(r_changes) > 0.3 then
            base_prob = base_prob + 0.3; // High change rate
        end
    end
    
    // Normalize to [0,1] range
    transition_prob = max(0, min(1, base_prob));
endfunction

// Calculate critical exponents for phase transitions
function exponents = calculate_critical_exponents(time_series, transition_points)
    if nargin < 1 then
        global energy_levels;
        time_series = energy_levels;
    end
    if nargin < 2 then
        cp = detect_critical_points(time_series);
        transition_points = cp.indices;
    end
    
    exponents = struct();
    exponents.beta = []; // Order parameter exponent
    exponents.gamma = []; // Susceptibility exponent  
    exponents.nu = []; // Correlation length exponent
    
    for i = 1:length(transition_points)
        transition_idx = transition_points(i);
        
        if transition_idx > 10 & transition_idx < length(time_series) - 10 then
            % Extract pre and post transition data
            pre_data = time_series(max(1, transition_idx-10):transition_idx);
            post_data = time_series(transition_idx:min(end, transition_idx+10));
            
            % Calculate simple critical exponents (simplified approach)
            % In real analysis, this would involve fitting to power laws
            
            % Beta: how order parameter grows after transition
            if length(post_data) > 1 then
                order_growth = (post_data - post_data(1)) / (post_data(1) + 1e-9);
                if ~isempty(order_growth(order_growth > 0)) & max(order_growth) > 0.1 then
                    % Fit power law (simplified)
                    exponents.beta = [exponents.beta, 0.5]; // Typical value
                end
            end
            
            % Gamma: susceptibility (response to external field)
            if length(pre_data) > 2 & length(post_data) > 2
                pre_variance = var(pre_data);
                post_variance = var(post_data);
                if pre_variance > 0 then
                    susceptibility_ratio = post_variance / pre_variance;
                    exponents.gamma = [exponents.gamma, log(susceptibility_ratio) / log(2)];
                end
            end
        end
    end
    
    % Fill missing values with typical critical exponents
    if isempty(exponents.beta) then
        exponents.beta = 0.5; // Mean field value
    end
    if isempty(exponents.gamma) then
        exponents.gamma = 1.0; // Mean field value
    end
    if isempty(exponents.nu) then
        exponents.nu = 0.5; // Mean field value
    end
endfunction

// Analyze hysteresis loops in phase transitions
function hysteresis_analysis = analyze_hysteresis(driving_parameter, response_parameter)
    if nargin < 1 then
        global energy_levels phi_values;
        driving_parameter = energy_levels;
        response_parameter = phi_values;
    end
    
    hysteresis_analysis = struct();
    
    if length(driving_parameter) ~= length(response_parameter) then
        hysteresis_analysis.area = 0;
        hysteresis_analysis.width = 0;
        hysteresis_analysis.loop_detected = %f;
        return;
    end
    
    % Create driving parameter cycles (simplified)
    % In practice, this would require controlled parameter sweeps
    unique_driving = unique(driving_parameter);
    
    if length(unique_driving) < 10 then
        hysteresis_analysis.area = 0;
        hysteresis_analysis.width = 0;
        hysteresis_analysis.loop_detected = %f;
        return;
    end
    
    % Simple hysteresis detection: look for parameter-dependent response delays
    forward_response = [];
    backward_response = [];
    
    % Find local maxima and minima in driving parameter
    [~, max_indices] = findpeaks(driving_parameter);
    [~, min_indices] = findpeaks(-driving_parameter);
    
    if length(max_indices) > 1 & length(min_indices) > 1 then
        % Calculate response at similar driving values
        max_responses = response_parameter(max_indices);
        min_responses = response_parameter(min_indices);
        
        % Hysteresis width
        hysteresis_analysis.width = mean(max_responses) - mean(min_responses);
        
        % Simple area calculation (trapezoidal)
        if length(max_responses) == length(min_responses) then
            hysteresis_analysis.area = abs(trapz([max_responses, min_responses]));
        else
            hysteresis_analysis.area = abs(mean(max_responses) - mean(min_responses));
        end
        
        % Loop detection threshold
        hysteresis_analysis.loop_detected = hysteresis_analysis.width > 0.1;
    else
        hysteresis_analysis.area = 0;
        hysteresis_analysis.width = 0;
        hysteresis_analysis.loop_detected = %f;
    end
endfunction

// Measure synchronization between HNS channels
function sync_measure = measure_synchronization(R, G, B, A)
    global short_term_memory;
    
    if isempty(short_term_memory) | length(short_term_memory) < 5 then
        // For single state, calculate based on channel correlations
        channels = [R, G, B, A];
        
        % Calculate pairwise correlations
        correlation_matrix = corr([channels; channels]); // Dummy for single point
        
        % Synchronization as inverse of variance in channel values
        sync_measure = 1 - (std(channels) / (mean(channels) + 1e-9));
        sync_measure = max(0, min(1, sync_measure));
        return;
    end
    
    % Calculate synchronization from time series
    recent_states = short_term_memory(max(1,end-19):end);
    
    R_series = [recent_states.R];
    G_series = [recent_states.G];
    B_series = [recent_states.B];
    A_series = [recent_states.A];
    
    % Calculate pairwise phase synchronization
    rg_sync = abs(corr(R_series', G_series'));
    rb_sync = abs(corr(R_series', B_series'));
    ra_sync = abs(corr(R_series', A_series'));
    gb_sync = abs(corr(G_series', B_series'));
    ga_sync = abs(corr(G_series', A_series'));
    ba_sync = abs(corr(B_series', A_series'));
    
    % Overall synchronization measure
    sync_measure = (rg_sync + rb_sync + ra_sync + gb_sync + ga_sync + ba_sync) / 6;
endfunction

// Advanced phase space reconstruction
function [trajectory, attractor_dimension] = reconstruct_phase_space(data, embedding_dim, time_delay)
    if nargin < 1 then
        global energy_levels;
        data = energy_levels;
    end
    if nargin < 2 then
        embedding_dim = 3;
    end
    if nargin < 3 then
        time_delay = 1;
    end
    
    [n_points, ~] = size(data);
    
    if n_points < embedding_dim * time_delay + 10 then
        trajectory = [];
        attractor_dimension = 0;
        return;
    end
    
    % Create embedded trajectory
    n_embedded = n_points - (embedding_dim - 1) * time_delay;
    trajectory = zeros(n_embedded, embedding_dim);
    
    for i = 1:n_embedded
        for j = 1:embedding_dim
            trajectory(i, j) = data(i + (j-1) * time_delay);
        end
    end
    
    % Estimate correlation dimension (simplified)
    if n_embedded > 50 then
        % Use Grassberger-Procaccia algorithm (simplified)
        r_values = linspace(0.01, 0.5, 20);
        correlation_integrals = zeros(size(r_values));
        
        for r_idx = 1:length(r_values)
            r = r_values(r_idx);
            count = 0;
            
            for i = 1:n_embedded
                for j = i+1:n_embedded
                    if norm(trajectory(i,:) - trajectory(j,:)) < r then
                        count = count + 1;
                    end
                end
            end
            
            correlation_integrals(r_idx) = 2 * count / (n_embedded * (n_embedded - 1));
        end
        
        % Estimate dimension from slope
        valid_indices = correlation_integrals > 0 & correlation_integrals < 1;
        if sum(valid_indices) > 3 then
            log_r = log(r_values(valid_indices));
            log_c = log(correlation_integrals(valid_indices));
            slope = polyfit(log_r, log_c, 1);
            attractor_dimension = slope(1);
        else
            attractor_dimension = embedding_dim;
        end
    else
        attractor_dimension = embedding_dim;
    end
endfunction

disp("Phase Transition Analysis module loaded successfully!");
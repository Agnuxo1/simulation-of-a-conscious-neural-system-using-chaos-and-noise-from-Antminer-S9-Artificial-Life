// ===============================================
// Modified Thermal and Power Model for BM1387 ASIC
// Supports firmware-configurable parameters for safe testing
// Demonstrates thermal management modifications without hardware risk
// ===============================================

`timescale 1ns/1ps

module thermal_model_modified (
    input wire clk,
    input wire reset_n,
    input wire [15:0] power_request,    // Requested power consumption (mW)
    output reg [7:0] temperature,       // Chip temperature (scaled 0-255 = 0-255°C)
    output reg [15:0] power_consumption, // Actual power consumption (mW)
    output reg throttle_request,        // Thermal throttling request
    input wire [15:0] hashes_per_second, // Hash rate for dynamic power
    // Firmware-configurable parameters
    input wire [7:0] firmware_temp_threshold_warning,   // Firmware warning threshold
    input wire [7:0] firmware_temp_threshold_critical,  // Firmware critical threshold
    input wire [15:0] firmware_power_limit_max          // Firmware power limit
);

    // Thermal parameters (original for comparison)
    parameter [15:0] AMBIENT_TEMP = 16'h1770; // 60°C in scaled format (60 * 25.6)
    parameter [15:0] T_JUNCTION_MAX = 16'h1F40; // 125°C maximum junction temp
    parameter [15:0] T_THROTTLE_START_ORIG = 16'h1B58; // 110°C throttle start (original)
    parameter [15:0] T_THROTTLE_FULL_ORIG = 16'h1F40;  // 125°C full throttle (original)
    
    // Thermal resistance and capacitance (simplified model)
    parameter [15:0] R_THETA_JA = 16'h00C8; // Thermal resistance 0.5°C/W (scaled)
    parameter [15:0] C_THERMAL = 16'h2710;  // Thermal capacitance (scaled)
    
    // Power model parameters (original values)
    parameter [15:0] POWER_IDLE_ORIG = 16'h02BC;      // 700mW idle power
    parameter [15:0] POWER_HASH_BASE_ORIG = 16'h0546; // 1350mW base hash power
    parameter [15:0] POWER_HASH_PER_HPS_ORIG = 16'h0001; // Additional power per 1000 H/s
    
    // Modified power model parameters (firmware configurable)
    parameter [15:0] POWER_IDLE_MOD = 16'h0258;      // 600mW idle power (reduced)
    parameter [15:0] POWER_HASH_BASE_MOD = 16'h04E2; // 1250mW base hash power (reduced)
    parameter [15:0] POWER_HASH_PER_HPS_MOD = 16'h0001; // Additional power per 1000 H/s
    
    // Internal temperature and power state
    reg [15:0] current_temp;
    reg [15:0] target_power;
    reg [15:0] thermal_inertia;
    reg [7:0] throttle_level; // 0-255 throttle level
    
    // Temperature calculation accumulators
    reg [31:0] temp_accumulator;
    reg [31:0] power_accumulator;
    
    // Hash rate smoothing for dynamic power
    reg [15:0] smoothed_hash_rate;
    reg [7:0] hash_smoothing_counter;
    
    // Thermal time constant simulation
    reg [15:0] thermal_step;
    reg [15:0] power_step;
    
    // Firmware adaptation parameters
    reg [15:0] adapted_power_limit;
    reg [7:0] firmware_safety_margin;
    reg [7:0] thermal_learning_rate;
    
    // Power consumption components (original calculation)
    wire [15:0] static_power_orig;
    wire [15:0] dynamic_power_orig;
    wire [15:0] total_power_orig;
    
    // Power consumption components (modified calculation)
    wire [15:0] static_power_mod;
    wire [15:0] dynamic_power_mod;
    wire [15:0] total_power_mod;
    
    // Calculate original static power (for comparison)
    assign static_power_orig = POWER_IDLE_ORIG;
    
    // Calculate original dynamic power based on hash rate
    assign dynamic_power_orig = POWER_HASH_BASE_ORIG + 
                               ((hashes_per_second * POWER_HASH_PER_HPS_ORIG) >> 10);
    
    // Total original power request
    assign total_power_orig = static_power_orig + dynamic_power_orig;
    
    // Calculate modified static power (firmware configurable)
    assign static_power_mod = POWER_IDLE_MOD;
    
    // Calculate modified dynamic power (firmware configurable)
    assign dynamic_power_mod = POWER_HASH_BASE_MOD + 
                              ((hashes_per_second * POWER_HASH_PER_HPS_MOD) >> 10);
    
    // Total modified power request
    assign total_power_mod = static_power_mod + dynamic_power_mod;
    
    // Temperature calculation and power management
    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            current_temp <= AMBIENT_TEMP;
            temperature <= 8'h00;
            target_power <= POWER_IDLE_MOD;
            power_consumption <= POWER_IDLE_MOD;
            throttle_request <= 1'b0;
            throttle_level <= 8'h00;
            thermal_inertia <= 16'h0000;
            temp_accumulator <= 32'h00000000;
            power_accumulator <= 32'h00000000;
            smoothed_hash_rate <= 16'h0000;
            hash_smoothing_counter <= 8'h00;
            thermal_step <= 16'h0000;
            power_step <= 16'h0000;
            
            // Initialize firmware adaptation parameters
            adapted_power_limit <= firmware_power_limit_max;
            firmware_safety_margin <= 8'h10; // 16°C safety margin
            thermal_learning_rate <= 8'h20; // Learning rate for adaptation
        end else begin
            // Smooth hash rate for dynamic power calculation
            if (hash_smoothing_counter == 8'hFF) begin
                smoothed_hash_rate <= hashes_per_second;
                hash_smoothing_counter <= 8'h00;
            end else begin
                hash_smoothing_counter <= hash_smoothing_counter + 1;
            end
            
            // Firmware-adaptive power limiting
            // Adapt power limit based on current thermal state
            if (current_temp > {firmware_temp_threshold_warning, 8'h00}) begin
                // Reduce power limit when approaching thermal limits
                adapted_power_limit <= firmware_power_limit_max - 
                                     ((current_temp - {firmware_temp_threshold_warning, 8'h00}) * thermal_learning_rate);
            end else begin
                // Gradually restore power limit when temperature is safe
                adapted_power_limit <= adapted_power_limit + (thermal_learning_rate >> 1);
                if (adapted_power_limit > firmware_power_limit_max) begin
                    adapted_power_limit <= firmware_power_limit_max;
                end
            end
            
            // Apply firmware-configurable thermal throttling
            if (current_temp > {firmware_temp_threshold_warning, 8'h00}) begin
                // Calculate throttle level based on firmware thresholds
                throttle_level <= ((current_temp - {firmware_temp_threshold_warning, 8'h00}) * 8'hFF) / 
                                 ({firmware_temp_threshold_critical, 8'h00} - {firmware_temp_threshold_warning, 8'h00});
                
                // Limit power consumption based on firmware power limit
                if (total_power_mod > adapted_power_limit) begin
                    target_power <= adapted_power_limit;
                end else begin
                    target_power <= total_power_mod;
                end
                throttle_request <= 1'b1;
            end else begin
                throttle_level <= 8'h00;
                target_power <= (total_power_mod > firmware_power_limit_max) ? firmware_power_limit_max : total_power_mod;
                throttle_request <= 1'b0;
            end
            
            // Enhanced thermal model with firmware safety features
            // Calculate temperature rise due to power consumption
            thermal_step <= (target_power * R_THETA_JA) >> 10; // Scaled division
            
            // Apply thermal inertia with firmware adaptation
            thermal_inertia <= (thermal_inertia * 7 + thermal_step) >> 3;
            
            // Calculate new temperature with firmware safety margin
            temp_accumulator <= AMBIENT_TEMP + thermal_inertia + {firmware_safety_margin, 8'h00};
            
            // Update temperature (with clamping based on firmware limits)
            if (temp_accumulator[15:8] > firmware_temp_threshold_critical) begin
                current_temp <= {firmware_temp_threshold_critical, 8'h00}; // Clamp to firmware limit
            end else begin
                current_temp <= temp_accumulator[15:0];
            end
            
            // Output temperature in 0-255 scale (0-255°C)
            temperature <= current_temp[15:8];
            
            // Update actual power consumption with firmware monitoring
            power_accumulator <= target_power + (temp_accumulator & 16'h00FF); // Add temp-dependent leakage
            
            // Apply firmware power limit
            if (power_accumulator[15:0] > adapted_power_limit) begin
                power_consumption <= adapted_power_limit;
            end else begin
                power_consumption <= power_accumulator[15:0];
            end
            
            // Emergency shutdown with firmware supervision
            if (current_temp >= {firmware_temp_threshold_critical, 8'h00}) begin
                target_power <= POWER_IDLE_MOD; // Reduce to idle power
                throttle_request <= 1'b1;
            end
        end
    end

endmodule
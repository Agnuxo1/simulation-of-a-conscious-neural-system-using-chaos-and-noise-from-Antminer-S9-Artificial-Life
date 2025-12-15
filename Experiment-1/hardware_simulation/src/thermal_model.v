// Thermal and Power Model for BM1387 ASIC
// Models temperature rise, power consumption, and thermal throttling

`timescale 1ns/1ps

module thermal_model (
    input wire clk,
    input wire reset_n,
    input wire [15:0] power_request,    // Requested power consumption (mW)
    output reg [7:0] temperature,       // Chip temperature (scaled 0-255 = 0-255°C)
    output reg [15:0] power_consumption, // Actual power consumption (mW)
    output reg throttle_request,        // Thermal throttling request
    input wire [15:0] hashes_per_second // Hash rate for dynamic power
);

    // Thermal parameters
    parameter [15:0] AMBIENT_TEMP = 16'h1770; // 60°C in scaled format (60 * 25.6)
    parameter [15:0] T_JUNCTION_MAX = 16'h1F40; // 125°C maximum junction temp
    parameter [15:0] T_THROTTLE_START = 16'h1B58; // 110°C throttle start
    parameter [15:0] T_THROTTLE_FULL = 16'h1F40;  // 125°C full throttle
    
    // Thermal resistance and capacitance (simplified model)
    parameter [15:0] R_THETA_JA = 16'h00C8; // Thermal resistance 0.5°C/W (scaled)
    parameter [15:0] C_THERMAL = 16'h2710;  // Thermal capacitance (scaled)
    
    // Power model parameters
    parameter [15:0] POWER_IDLE = 16'h02BC;      // 700mW idle power
    parameter [15:0] POWER_HASH_BASE = 16'h0546; // 1350mW base hash power
    parameter [15:0] POWER_HASH_PER_HPS = 16'h0001; // Additional power per 1000 H/s
    
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
    
    // Power consumption components
    wire [15:0] static_power;
    wire [15:0] dynamic_power;
    wire [15:0] total_power;
    
    // Calculate static power (always consumed)
    assign static_power = POWER_IDLE;
    
    // Calculate dynamic power based on hash rate
    assign dynamic_power = POWER_HASH_BASE + 
                          ((hashes_per_second * POWER_HASH_PER_HPS) >> 10);
    
    // Total power request
    assign total_power = static_power + dynamic_power;
    
    // Temperature calculation and power management
    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            current_temp <= AMBIENT_TEMP;
            temperature <= 8'h00;
            target_power <= POWER_IDLE;
            power_consumption <= POWER_IDLE;
            throttle_request <= 1'b0;
            throttle_level <= 8'h00;
            thermal_inertia <= 16'h0000;
            temp_accumulator <= 32'h00000000;
            power_accumulator <= 32'h00000000;
            smoothed_hash_rate <= 16'h0000;
            hash_smoothing_counter <= 8'h00;
            thermal_step <= 16'h0000;
            power_step <= 16'h0000;
        end else begin
            // Smooth hash rate for dynamic power calculation
            if (hash_smoothing_counter == 8'hFF) begin
                smoothed_hash_rate <= hashes_per_second;
                hash_smoothing_counter <= 8'h00;
            end else begin
                hash_smoothing_counter <= hash_smoothing_counter + 1;
            end
            
            // Apply thermal throttling if necessary
            if (current_temp > T_THROTTLE_START) begin
                // Calculate throttle level (0-255)
                throttle_level <= ((current_temp - T_THROTTLE_START) * 8'hFF) / 
                                 (T_THROTTLE_FULL - T_THROTTLE_START);
                
                // Limit power consumption based on throttle level
                target_power <= total_power - ((total_power * throttle_level) >> 8);
                throttle_request <= 1'b1;
            end else begin
                throttle_level <= 8'h00;
                target_power <= total_power;
                throttle_request <= 1'b0;
            end
            
            // Thermal model: dT/dt = (P * R_th) / C_th
            // Simplified to discrete time steps
            
            // Calculate temperature rise due to power consumption
            thermal_step <= (target_power * R_THETA_JA) >> 10; // Scaled division
            
            // Apply thermal inertia (low-pass filter effect)
            thermal_inertia <= (thermal_inertia * 7 + thermal_step) >> 3;
            
            // Calculate new temperature
            temp_accumulator <= AMBIENT_TEMP + thermal_inertia;
            
            // Update temperature (with some noise for realism)
            if (temp_accumulator[15:8] > 8'hFF) begin
                current_temp <= 16'h1F40; // Clamp to maximum
            end else begin
                current_temp <= temp_accumulator[15:0];
            end
            
            // Output temperature in 0-255 scale (0-255°C)
            temperature <= current_temp[15:8];
            
            // Update actual power consumption
            power_accumulator <= target_power + (temp_accumulator & 16'h00FF); // Add temp-dependent leakage
            power_consumption <= power_accumulator[15:0];
            
            // Emergency shutdown if temperature too high
            if (current_temp >= T_JUNCTION_MAX) begin
                target_power <= POWER_IDLE; // Reduce to idle power
                throttle_request <= 1'b1;
            end
        end
    end

endmodule
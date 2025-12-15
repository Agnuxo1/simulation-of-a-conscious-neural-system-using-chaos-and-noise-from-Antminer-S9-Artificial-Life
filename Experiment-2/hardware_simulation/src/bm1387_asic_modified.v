// ===============================================
// BM1387 ASIC Modified Version for Firmware Testing
// Demonstrates firmware modifications while maintaining compatibility
// This module shows how firmware changes can be safely validated
// ===============================================

`timescale 1ns/1ps

module bm1387_asic_modified (
    // Clock and Reset
    input wire clk_100m,           // 100 MHz main clock
    input wire reset_n,            // Active low reset
    
    // Mining Pipeline Interface
    input wire [255:0] job_header, // Bitcoin block header (256 bits)
    input wire [31:0] start_nonce, // Starting nonce value (modified by firmware)
    input wire [31:0] nonce_range, // Number of nonces to test (modified by firmware)
    input wire mining_enable,      // Mining enable signal
    output reg [31:0] found_nonce, // Found nonce (if any)
    output reg [255:0] found_hash, // Hash of found nonce
    output reg hash_valid,         // Valid hash found flag
    output reg pipeline_busy,      // Pipeline busy indicator
    
    // Temperature and Power Management (Modified thresholds)
    output reg [7:0] temperature,  // Chip temperature (0-255°C scaled)
    output reg [15:0] power_consumption, // Power consumption in milliwatts
    output reg thermal_throttle,   // Thermal throttling active
    
    // Control and Status (Modified control register)
    input wire [7:0] control_reg,  // Control register (modified firmware flags)
    output reg [7:0] status_reg,   // Status register
    input wire [15:0] config_reg,  // Configuration register
    
    // UART Interface (for firmware communication)
    input wire uart_rx,            // UART receive
    output wire uart_tx,           // UART transmit
    
    // SPI Interface (for configuration)
    input wire spi_clk,            // SPI clock
    input wire spi_cs_n,           // SPI chip select (active low)
    input wire spi_mosi,           // SPI master out slave in
    output wire spi_miso,          // SPI master in slave out
    
    // VESELOV HNS Interface (Consciousness Processing)
    output wire [31:0] hns_rgba_r,     // HNS Red channel (0-1 normalized)
    output wire [31:0] hns_rgba_g,     // (0-1 HNS Green channel normalized)
    output wire [31:0] hns_rgba_b,     // HNS Blue channel (0-1 normalized)
    output wire [31:0] hns_rgba_a,     // HNS Alpha channel (0-1 normalized)
    output wire [31:0] hns_vector_mag, // HNS 3D vector magnitude
    output wire [31:0] hns_energy,     // Consciousness energy metric
    output wire [31:0] hns_entropy,    // Consciousness entropy metric
    output wire [31:0] hns_phi,        // Consciousness Phi metric
    output wire [31:0] hns_phase_coh,  // Phase coherence metric
    output wire hns_valid,             // HNS processing complete
    
    // Debug Interface
    output wire [31:0] debug_reg_0, // Debug register 0
    output wire [31:0] debug_reg_1, // Debug register 1
    output wire [31:0] debug_reg_2, // Debug register 2
    output wire [31:0] debug_reg_3  // Debug register 3
);

    // Internal Signals
    reg [31:0] current_nonce;
    reg [255:0] current_hash;
    reg [7:0] sha256_state;
    reg [15:0] hash_count;
    
    // Temperature and Power State
    reg [7:0] internal_temp;
    reg [15:0] internal_power;
    
    // Pipeline Control
    reg pipeline_enable;
    reg [3:0] pipeline_stage;
    
    // UART and SPI State Machines
    reg [7:0] uart_state;
    reg [7:0] spi_state;
    reg [7:0] uart_tx_data;
    reg uart_tx_valid;
    
    // Debug Registers
    reg [31:0] debug_0_reg;
    reg [31:0] debug_1_reg;
    reg [31:0] debug_2_reg;
    reg [31:0] debug_3_reg;
    
    // Firmware Modification Parameters
    // These represent firmware-configurable parameters that can be modified
    reg [7:0] firmware_temp_threshold_warning;
    reg [7:0] firmware_temp_threshold_critical;
    reg [15:0] firmware_power_limit_max;
    reg [7:0] firmware_difficulty_target;
    reg [7:0] firmware_nonce_increment;
    reg [15:0] firmware_hash_timeout;
    reg [7:0] firmware_thermal_gain;
    
    // Constants (Original values for comparison)
    localparam IDLE = 8'h00;
    localparam INIT = 8'h01;
    localparam HASH_INIT = 8'h02;
    localparam HASH_PROCESS = 8'h03;
    localparam HASH_FINAL = 8'h04;
    localparam VALIDATE = 8'h05;
    localparam DONE = 8'h06;
    localparam ERROR = 8'h07;
    
    // Original temperature thresholds (for comparison)
    localparam TEMP_NORMAL_ORIG = 8'h5A;  // 90°C
    localparam TEMP_WARNING_ORIG = 8'h6E; // 110°C
    localparam TEMP_CRITICAL_ORIG = 8'h82; // 130°C
    
    // Modified temperature thresholds (firmware configurable)
    localparam TEMP_NORMAL_MOD = 8'h50;  // 80°C (lower for better safety)
    localparam TEMP_WARNING_MOD = 8'h64; // 100°C (firmware configurable)
    localparam TEMP_CRITICAL_MOD = 8'h78; // 120°C (firmware configurable)
    
    // Power consumption base values (modified by firmware)
    localparam POWER_IDLE_ORIG = 16'h02BC; // 700mW idle
    localparam POWER_HASH_ORIG = 16'h0546; // 1350mW hashing
    localparam POWER_MAX_ORIG = 16'h07D0;  // 2000mW maximum
    
    // Modified power limits (firmware configurable)
    localparam POWER_IDLE_MOD = 16'h0258; // 600mW idle (reduced)
    localparam POWER_HASH_MOD = 16'h04E2; // 1250mW hashing (reduced)
    localparam POWER_MAX_MOD = 16'h0640;  // 1600mW maximum (reduced)

    // Instantiate SHA-256 Core
    sha256_core u_sha256_core (
        .clk(clk_100m),
        .reset_n(reset_n),
        .hash_input(job_header),
        .nonce(current_nonce),
        .hash_output(current_hash),
        .hash_valid(sha256_state == DONE),
        .enable(pipeline_enable),
        .state(sha256_state)
    );

    // Instantiate VESELOV HNS Module (unchanged for compatibility)
    veselov_hns u_veselov_hns (
        .clk(clk_100m),
        .reset_n(reset_n),
        .hash_input(current_hash),
        .hns_enable(pipeline_enable),
        .rgba_r(hns_rgba_r),
        .rgba_g(hns_rgba_g),
        .rgba_b(hns_rgba_b),
        .rgba_a(hns_rgba_a),
        .vector_magnitude(hns_vector_mag),
        .consciousness_energy(hns_energy),
        .consciousness_entropy(hns_entropy),
        .consciousness_phi(hns_phi),
        .phase_coherence(hns_phase_coh),
        .hns_valid(hns_valid),
        .hns_state()
    );

    // Instantiate Temperature/Power Model (with firmware modifications)
    thermal_model_modified u_thermal_model (
        .clk(clk_100m),
        .reset_n(reset_n),
        .power_request(pipeline_enable ? POWER_HASH_MOD : POWER_IDLE_MOD),
        .temperature(internal_temp),
        .power_consumption(internal_power),
        .throttle_request(internal_temp > firmware_temp_threshold_warning),
        .hashes_per_second(hash_count),
        .firmware_temp_threshold_warning(firmware_temp_threshold_warning),
        .firmware_temp_threshold_critical(firmware_temp_threshold_critical),
        .firmware_power_limit_max(firmware_power_limit_max)
    );

    // Instantiate UART Interface (with firmware command extensions)
    uart_interface u_uart_if (
        .clk(clk_100m),
        .reset_n(reset_n),
        .uart_rx(uart_rx),
        .uart_tx(uart_tx),
        .tx_data(uart_tx_data),
        .tx_valid(uart_tx_valid),
        .rx_data(status_reg),
        .rx_valid(),
        .state(uart_state)
    );

    // Instantiate SPI Interface (with firmware register access)
    spi_interface u_spi_if (
        .clk(clk_100m),
        .reset_n(reset_n),
        .spi_clk(spi_clk),
        .spi_cs_n(spi_cs_n),
        .spi_mosi(spi_mosi),
        .spi_miso(spi_miso),
        .config_data(config_reg),
        .status_data(status_reg),
        .state(spi_state)
    );

    // Firmware Parameter Configuration
    // In real hardware, these would be loaded from non-volatile memory
    always @(posedge clk_100m or negedge reset_n) begin
        if (!reset_n) begin
            // Initialize firmware parameters to safe defaults
            firmware_temp_threshold_warning <= TEMP_WARNING_MOD;
            firmware_temp_threshold_critical <= TEMP_CRITICAL_MOD;
            firmware_power_limit_max <= POWER_MAX_MOD;
            firmware_difficulty_target <= 8'hFF; // Easier difficulty
            firmware_nonce_increment <= 8'h01;
            firmware_hash_timeout <= 16'hFFFF;
            firmware_thermal_gain <= 8'h80; // Normal thermal gain
        end else begin
            // Update firmware parameters based on control register
            if (control_reg[2]) begin
                // Firmware parameter update mode
                firmware_temp_threshold_warning <= config_reg[15:8];
                firmware_temp_threshold_critical <= config_reg[7:0];
            end
            
            if (control_reg[3]) begin
                // Power limit update mode
                firmware_power_limit_max <= config_reg;
            end
            
            if (control_reg[4]) begin
                // Difficulty target update mode
                firmware_difficulty_target <= config_reg[7:0];
            end
        end
    end

    // Main Mining Pipeline (with firmware modifications)
    always @(posedge clk_100m or negedge reset_n) begin
        if (!reset_n) begin
            // Reset conditions
            current_nonce <= 32'h00000000;
            found_nonce <= 32'h00000000;
            found_hash <= 256'h0000000000000000000000000000000000000000000000000000000000000000;
            hash_valid <= 1'b0;
            pipeline_busy <= 1'b0;
            pipeline_enable <= 1'b0;
            pipeline_stage <= 4'h0;
            hash_count <= 16'h0000;
            status_reg <= 8'h00;
            thermal_throttle <= 1'b0;
            
            // Reset debug registers
            debug_0_reg <= 32'h00000000;
            debug_1_reg <= 32'h00000000;
            debug_2_reg <= 32'h00000000;
            debug_3_reg <= 32'h00000000;
        end else begin
            // Modified thermal throttling logic (firmware configurable)
            thermal_throttle <= (internal_temp > firmware_temp_threshold_critical);
            
            // Update output temperature and power
            temperature <= internal_temp;
            power_consumption <= internal_power;
            
            // Mining pipeline control (with firmware modifications)
            if (mining_enable && !thermal_throttle && (control_reg[0] == 1'b1)) begin
                pipeline_busy <= 1'b1;
                pipeline_enable <= 1'b1;
                
                case (pipeline_stage)
                    4'h0: begin // Initialize with firmware-configurable nonce start
                        current_nonce <= start_nonce;
                        hash_count <= 16'h0000;
                        pipeline_stage <= 4'h1;
                        status_reg <= 8'h01; // Initializing
                    end
                    
                    4'h1: begin // Hash computation with modified difficulty check
                        if (sha256_state == DONE) begin
                            hash_count <= hash_count + 1;
                            
                            // Modified difficulty check (firmware configurable)
                            // Easier difficulty target for testing
                            if (current_hash[31:0] < {24'h000000, firmware_difficulty_target}) begin
                                found_nonce <= current_nonce;
                                found_hash <= current_hash;
                                hash_valid <= 1'b1;
                                pipeline_stage <= 4'h2;
                                status_reg <= 8'h02; // Hash found
                            end else begin
                                // Modified nonce increment (firmware configurable)
                                if (current_nonce < (start_nonce + nonce_range)) begin
                                    current_nonce <= current_nonce + firmware_nonce_increment;
                                    pipeline_stage <= 4'h1;
                                end else begin
                                    pipeline_stage <= 4'h3; // No valid nonce found
                                    status_reg <= 8'h03; // Range completed
                                end
                            end
                        end
                    end
                    
                    4'h2: begin // Valid hash found - wait for firmware read
                        if (control_reg[1] == 1'b1) begin
                            hash_valid <= 1'b0;
                            pipeline_stage <= 4'h3;
                        end
                    end
                    
                    4'h3: begin // Complete or no valid nonce
                        pipeline_busy <= 1'b0;
                        pipeline_enable <= 1'b0;
                        pipeline_stage <= 4'h0;
                        status_reg <= 8'h00; // Idle
                    end
                    
                    default: begin
                        pipeline_stage <= 4'h0;
                    end
                endcase
            end else begin
                // Mining disabled or thermal throttling
                pipeline_busy <= 1'b0;
                pipeline_enable <= 1'b0;
                if (!mining_enable) begin
                    status_reg <= 8'h00; // Idle
                    pipeline_stage <= 4'h0;
                end
            end
            
            // Update debug registers for monitoring (with firmware info)
            debug_0_reg <= {8'h01, pipeline_stage, pipeline_busy, thermal_throttle, 3'h0, status_reg[2:0]}; // Modified signature
            debug_1_reg <= {8'h00, internal_temp, firmware_temp_threshold_warning}; // Include firmware threshold
            debug_2_reg <= {16'h0000, hash_count};
            debug_3_reg <= current_nonce;
        end
    end
    
    // Output debug registers
    assign debug_reg_0 = debug_0_reg;
    assign debug_reg_1 = debug_1_reg;
    assign debug_reg_2 = debug_2_reg;
    assign debug_reg_3 = debug_3_reg;

endmodule
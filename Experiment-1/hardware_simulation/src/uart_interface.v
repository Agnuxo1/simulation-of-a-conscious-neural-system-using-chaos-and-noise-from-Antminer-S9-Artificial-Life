// UART Interface for BM1387 ASIC
// Simulates firmware communication interface

`timescale 1ns/1ps

module uart_interface (
    input wire clk,
    input wire reset_n,
    input wire uart_rx,              // UART receive input
    output reg uart_tx,              // UART transmit output
    input wire [7:0] tx_data,        // Data to transmit
    input wire tx_valid,             // Transmit valid signal
    output reg [7:0] rx_data,        // Received data
    output reg rx_valid,             // Receive valid signal
    output reg [7:0] state           // Internal state for debugging
);

    // UART parameters
    parameter CLOCK_FREQ = 100_000_000; // 100 MHz system clock
    parameter BAUD_RATE = 115200;       // 115200 baud UART
    parameter BAUD_DIVISOR = CLOCK_FREQ / (BAUD_RATE * 16);
    
    // State machine states
    localparam IDLE = 8'h00;
    localparam RX_START_BIT = 8'h01;
    localparam RX_DATA_BITS = 8'h02;
    localparam RX_STOP_BIT = 8'h03;
    localparam TX_START_BIT = 8'h04;
    localparam TX_DATA_BITS = 8'h05;
    localparam TX_STOP_BIT = 8'h06;
    localparam TX_COMPLETE = 8'h07;
    
    // RX shift register and state
    reg [7:0] rx_shift_reg;
    reg [3:0] rx_bit_count;
    reg [15:0] rx_baud_counter;
    reg rx_sample_enable;
    reg rx_data_valid;
    
    // TX shift register and state
    reg [7:0] tx_shift_reg;
    reg [3:0] tx_bit_count;
    reg [15:0] tx_baud_counter;
    reg tx_busy;
    reg tx_output;
    
    // Baud rate generation
    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            rx_baud_counter <= 16'h0000;
            tx_baud_counter <= 16'h0000;
            rx_sample_enable <= 1'b0;
        end else begin
            // RX baud counter
            if (rx_baud_counter == BAUD_DIVISOR - 1) begin
                rx_baud_counter <= 16'h0000;
                rx_sample_enable <= 1'b1;
            end else begin
                rx_baud_counter <= rx_baud_counter + 1;
                rx_sample_enable <= 1'b0;
            end
            
            // TX baud counter
            if (tx_busy) begin
                if (tx_baud_counter == BAUD_DIVISOR - 1) begin
                    tx_baud_counter <= 16'h0000;
                end else begin
                    tx_baud_counter <= tx_baud_counter + 1;
                end
            end else begin
                tx_baud_counter <= 16'h0000;
            end
        end
    end
    
    // RX state machine
    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            state <= IDLE;
            rx_data <= 8'h00;
            rx_valid <= 1'b0;
            rx_shift_reg <= 8'h00;
            rx_bit_count <= 4'h0;
            rx_data_valid <= 1'b0;
        end else begin
            case (state)
                IDLE: begin
                    rx_valid <= 1'b0;
                    rx_data_valid <= 1'b0;
                    if (!uart_rx) begin
                        state <= RX_START_BIT;
                        rx_bit_count <= 4'h0;
                    end
                end
                
                RX_START_BIT: begin
                    if (rx_sample_enable) begin
                        if (!uart_rx) begin
                            state <= RX_DATA_BITS;
                            rx_bit_count <= 4'h0;
                            rx_shift_reg <= 8'h00;
                        end else begin
                            state <= IDLE; // False start bit
                        end
                    end
                end
                
                RX_DATA_BITS: begin
                    if (rx_sample_enable) begin
                        rx_shift_reg <= {uart_rx, rx_shift_reg[7:1]};
                        rx_bit_count <= rx_bit_count + 1;
                        if (rx_bit_count == 4'h7) begin
                            state <= RX_STOP_BIT;
                        end
                    end
                end
                
                RX_STOP_BIT: begin
                    if (rx_sample_enable) begin
                        if (uart_rx) begin
                            rx_data <= rx_shift_reg;
                            rx_valid <= 1'b1;
                            rx_data_valid <= 1'b1;
                            state <= IDLE;
                        end else begin
                            state <= IDLE; // Framing error
                        end
                    end
                end
                
                default: begin
                    state <= IDLE;
                end
            endcase
        end
    end
    
    // TX state machine
    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            uart_tx <= 1'b1; // UART idle high
            tx_busy <= 1'b0;
            tx_output <= 1'b1;
            tx_shift_reg <= 8'h00;
            tx_bit_count <= 4'h0;
        end else begin
            case (state)
                IDLE: begin
                    uart_tx <= 1'b1;
                    tx_output <= 1'b1;
                    tx_busy <= 1'b0;
                    if (tx_valid) begin
                        tx_shift_reg <= tx_data;
                        tx_bit_count <= 4'h0;
                        state <= TX_START_BIT;
                    end
                end
                
                TX_START_BIT: begin
                    tx_busy <= 1'b1;
                    tx_output <= 1'b0; // Start bit low
                    if (tx_baud_counter == BAUD_DIVISOR - 1) begin
                        state <= TX_DATA_BITS;
                        tx_output <= tx_shift_reg[0];
                    end
                end
                
                TX_DATA_BITS: begin
                    tx_output <= tx_shift_reg[tx_bit_count];
                    if (tx_baud_counter == BAUD_DIVISOR - 1) begin
                        tx_bit_count <= tx_bit_count + 1;
                        if (tx_bit_count == 4'h7) begin
                            state <= TX_STOP_BIT;
                            tx_output <= 1'b1; // Prepare for stop bit
                        end
                    end
                end
                
                TX_STOP_BIT: begin
                    tx_output <= 1'b1; // Stop bit high
                    if (tx_baud_counter == BAUD_DIVISOR - 1) begin
                        state <= TX_COMPLETE;
                    end
                end
                
                TX_COMPLETE: begin
                    uart_tx <= tx_output;
                    state <= IDLE;
                end
                
                default: begin
                    state <= IDLE;
                end
            endcase
            
            // Update UART TX output
            if (state != TX_COMPLETE) begin
                uart_tx <= tx_output;
            end
        end
    end
    
    // Handle received data (process firmware commands)
    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            // Reset command processing state
        end else begin
            if (rx_valid) begin
                // Process received commands
                case (rx_data)
                    8'h01: begin // Get status command
                        // Status response will be sent via TX
                    end
                    8'h02: begin // Get temperature command
                        // Temperature response
                    end
                    8'h03: begin // Get hash rate command
                        // Hash rate response
                    end
                    8'h04: begin // Reset command
                        // Reset ASIC
                    end
                    default: begin
                        // Unknown command
                    end
                endcase
                rx_valid <= 1'b0; // Clear valid flag
            end
        end
    end

endmodule
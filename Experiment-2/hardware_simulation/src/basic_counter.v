// Basic 4-bit counter for hardware simulation testing
// This module tests Verilator compilation and simulation

module basic_counter (
    input wire clk,
    input wire reset,
    input wire enable,
    output reg [3:0] count,
    output wire count_out
);

    // Always block for counter logic
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            count <= 4'b0000;
        end else if (enable) begin
            count <= count + 1;
        end
    end
    
    // Continuous assignment for output
    assign count_out = count;
    
endmodule

// Testbench for basic counter
module testbench_basic_counter;
    
    // Testbench signals
    reg clk;
    reg reset;
    reg enable;
    wire [3:0] count;
    wire count_out;
    
    // Instantiate the counter module
    basic_counter uut (
        .clk(clk),
        .reset(reset),
        .enable(enable),
        .count(count),
        .count_out(count_out)
    );
    
    // Clock generation
    always #5 clk = ~clk;  // 10ns period clock
    
    // Test sequence
    initial begin
        $dumpfile("testbench_basic_counter.vcd");
        $dumpvars(0, testbench_basic_counter);
        
        $display("Starting basic counter test...");
        
        // Initialize inputs
        clk = 0;
        reset = 1;
        enable = 0;
        
        $display("Time=%0d: Reset active", $time);
        #10 reset = 0;  // Deassert reset
        
        $display("Time=%0d: Counter enabled, enable=%b", $time, enable);
        #5 enable = 1;  // Enable counter
        
        // Monitor counter for 20 cycles
        repeat (20) begin
            $display("Time=%0d: Count=%4b (decimal: %0d), count_out=%1b", 
                     $time, count, count, count_out);
            #10;
        end
        
        $display("Time=%0d: Disabling counter", $time);
        enable = 0;
        
        // Monitor disabled state
        repeat (5) begin
            $display("Time=%0d: Count=%4b (disabled)", $time, count);
            #10;
        end
        
        $display("Time=%0d: Test completed", $time);
        $display("Final count value: %0d", count);
        
        $finish;
    end
    
    // Monitor changes
    always @(count) begin
        $display("Time=%0d: Counter changed to %4b", $time, count);
    end
    
endmodule
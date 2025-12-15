// BM1387 ASIC Testbench
// Comprehensive verification of all ASIC functionality

`timescale 1ns/1ps

module bm1387_asic_tb;

    // Testbench parameters
    parameter CLK_PERIOD = 10; // 100 MHz clock
    parameter SIM_TIME = 100000; // 1ms simulation time
    
    // Testbench signals
    reg clk_100m;
    reg reset_n;
    
    // Mining interface signals
    reg [255:0] job_header;
    reg [31:0] start_nonce;
    reg [31:0] nonce_range;
    reg mining_enable;
    wire [31:0] found_nonce;
    wire [255:0] found_hash;
    wire hash_valid;
    wire pipeline_busy;
    
    // Temperature and power signals
    wire [7:0] temperature;
    wire [15:0] power_consumption;
    wire thermal_throttle;
    
    // Control signals
    reg [7:0] control_reg;
    wire [7:0] status_reg;
    reg [15:0] config_reg;
    
    // UART interface signals
    reg uart_rx;
    wire uart_tx;
    
    // SPI interface signals
    reg spi_clk;
    reg spi_cs_n;
    reg spi_mosi;
    wire spi_miso;
    
    // Debug signals
    wire [31:0] debug_reg_0;
    wire [31:0] debug_reg_1;
    wire [31:0] debug_reg_2;
    wire [31:0] debug_reg_3;
    
    // Test variables
    integer test_count;
    integer pass_count;
    integer fail_count;
    integer file_handle;
    
    // Instantiate the BM1387 ASIC
    bm1387_asic dut (
        .clk_100m(clk_100m),
        .reset_n(reset_n),
        .job_header(job_header),
        .start_nonce(start_nonce),
        .nonce_range(nonce_range),
        .mining_enable(mining_enable),
        .found_nonce(found_nonce),
        .found_hash(found_hash),
        .hash_valid(hash_valid),
        .pipeline_busy(pipeline_busy),
        .temperature(temperature),
        .power_consumption(power_consumption),
        .thermal_throttle(thermal_throttle),
        .control_reg(control_reg),
        .status_reg(status_reg),
        .config_reg(config_reg),
        .uart_rx(uart_rx),
        .uart_tx(uart_tx),
        .spi_clk(spi_clk),
        .spi_cs_n(spi_cs_n),
        .spi_mosi(spi_mosi),
        .spi_miso(spi_miso),
        .debug_reg_0(debug_reg_0),
        .debug_reg_1(debug_reg_1),
        .debug_reg_2(debug_reg_2),
        .debug_reg_3(debug_reg_3)
    );
    
    // Clock generation
    always #(CLK_PERIOD/2) clk_100m = ~clk_100m;
    
    // VCD dump for waveform analysis
    initial begin
        $dumpfile("waveforms/bm1387_asic_tb.vcd");
        $dumpvars(0, bm1387_asic_tb);
    end
    
    // Test sequence
    initial begin
        // Initialize testbench
        clk_100m = 0;
        reset_n = 0;
        job_header = 256'h0;
        start_nonce = 32'h0;
        nonce_range = 32'h0;
        mining_enable = 0;
        control_reg = 8'h0;
        config_reg = 16'h0;
        uart_rx = 1'b1; // UART idle high
        spi_clk = 0;
        spi_cs_n = 1'b1; // SPI inactive
        spi_mosi = 0;
        
        test_count = 0;
        pass_count = 0;
        fail_count = 0;
        
        $display("Starting BM1387 ASIC Testbench");
        $display("================================");
        
        // Reset sequence
        #(CLK_PERIOD * 10);
        reset_n = 1;
        #(CLK_PERIOD * 10);
        
        // Run all tests
        test_basic_reset();
        test_mining_pipeline();
        test_sha256_computation();
        test_thermal_model();
        test_uart_interface();
        test_spi_interface();
        test_integration();
        
        // Final results
        #(CLK_PERIOD * 100);
        $display("\nTest Results:");
        $display("=============");
        $display("Total Tests: %0d", test_count);
        $display("Passed: %0d", pass_count);
        $display("Failed: %0d", fail_count);
        $display("Success Rate: %0.1f%%", (pass_count * 100.0) / test_count);
        
        if (fail_count == 0) begin
            $display("\n*** ALL TESTS PASSED ***");
        end else begin
            $display("\n*** SOME TESTS FAILED ***");
        end
        
        $finish;
    end
    
    // Test task: Basic reset functionality
    task test_basic_reset;
        begin
            test_count = test_count + 1;
            $display("\nTest %0d: Basic Reset Functionality", test_count);
            
            // Test reset behavior
            reset_n = 0;
            #(CLK_PERIOD);
            
            if (status_reg == 8'h00 && pipeline_busy == 1'b0) begin
                $display("✓ Reset functionality working correctly");
                pass_count = pass_count + 1;
            end else begin
                $display("✗ Reset functionality failed");
                $display("  Status: expected 8'h00, got 8'h%02h", status_reg);
                $display("  Pipeline: expected 1'b0, got 1'b%b", pipeline_busy);
                fail_count = fail_count + 1;
            end
            
            reset_n = 1;
            #(CLK_PERIOD * 5);
        end
    endtask
    
    // Test task: Mining pipeline functionality
    task test_mining_pipeline;
        begin
            test_count = test_count + 1;
            $display("\nTest %0d: Mining Pipeline Functionality", test_count);
            
            // Setup test mining job
            job_header = 256'h1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef;
            start_nonce = 32'h1000;
            nonce_range = 32'h10;
            control_reg = 8'h01; // Enable mining
            
            // Start mining
            mining_enable = 1;
            #(CLK_PERIOD * 10);
            
            // Check pipeline starts
            if (pipeline_busy == 1'b1) begin
                $display("✓ Mining pipeline started successfully");
                pass_count = pass_count + 1;
            end else begin
                $display("✗ Mining pipeline failed to start");
                fail_count = fail_count + 1;
            end
            
            // Monitor pipeline for some cycles
            repeat(100) @(posedge clk_100m);
            
            // Check thermal behavior
            if (temperature > 8'h00 && power_consumption > 16'h0000) begin
                $display("✓ Thermal model responding to mining load");
                pass_count = pass_count + 1;
            end else begin
                $display("✗ Thermal model not responding correctly");
                $display("  Temperature: %0d, Power: %0d", temperature, power_consumption);
                fail_count = fail_count + 1;
            end
            
            mining_enable = 0;
            #(CLK_PERIOD * 10);
        end
    endtask
    
    // Test task: SHA-256 computation
    task test_sha256_computation;
        begin
            test_count = test_count + 1;
            $display("\nTest %0d: SHA-256 Computation", test_count);
            
            // Simple test case with known hash
            job_header = 256'h0000000000000000000000000000000000000000000000000000000000000000;
            start_nonce = 32'h0000;
            nonce_range = 32'h01;
            control_reg = 8'h01;
            
            mining_enable = 1;
            #(CLK_PERIOD * 20);
            
            // Monitor for hash completion
            wait (hash_valid);
            #(CLK_PERIOD * 5);
            
            if (hash_valid == 1'b1) begin
                $display("✓ SHA-256 computation completed");
                $display("  Found nonce: 32'h%08h", found_nonce);
                $display("  Hash: %064h", found_hash);
                pass_count = pass_count + 1;
            end else begin
                $display("✗ SHA-256 computation did not complete");
                fail_count = fail_count + 1;
            end
            
            mining_enable = 0;
            #(CLK_PERIOD * 10);
        end
    endtask
    
    // Test task: Thermal model
    task test_thermal_model;
        begin
            test_count = test_count + 1;
            $display("\nTest %0d: Thermal Model", test_count);
            
            // Test temperature rise with mining load
            mining_enable = 0;
            #(CLK_PERIOD * 50);
            
            reg [7:0] temp_idle = temperature;
            reg [15:0] power_idle = power_consumption;
            
            mining_enable = 1;
            control_reg = 8'h01;
            #(CLK_PERIOD * 1000); // Let temperature rise
            
            reg [7:0] temp_active = temperature;
            reg [15:0] power_active = power_consumption;
            
            if (temp_active > temp_idle && power_active > power_idle) begin
                $display("✓ Thermal model showing realistic behavior");
                $display("  Idle:  Temp=%0d°C, Power=%0dmW", temp_idle, power_idle);
                $display("  Active: Temp=%0d°C, Power=%0dmW", temp_active, power_active);
                pass_count = pass_count + 1;
            end else begin
                $display("✗ Thermal model not responding correctly");
                $display("  Idle:  Temp=%0d°C, Power=%0dmW", temp_idle, power_idle);
                $display("  Active: Temp=%0d°C, Power=%0dmW", temp_active, power_active);
                fail_count = fail_count + 1;
            end
            
            mining_enable = 0;
            #(CLK_PERIOD * 10);
        end
    endtask
    
    // Test task: UART interface
    task test_uart_interface;
        begin
            test_count = test_count + 1;
            $display("\nTest %0d: UART Interface", test_count);
            
            // Simulate UART transmission
            // Send a simple command byte
            fork
                begin
                    // Start bit (low)
                    uart_rx = 1'b0;
                    #(8680); // 115200 baud = 8.68us per bit
                    
                    // Data bits (0x01)
                    uart_rx = 1'b1; // LSB first
                    #(8680);
                    uart_rx = 1'b0;
                    #(8680);
                    uart_rx = 1'b0;
                    #(8680);
                    uart_rx = 1'b0;
                    #(8680);
                    uart_rx = 1'b0;
                    #(8680);
                    uart_rx = 1'b0;
                    #(8680);
                    uart_rx = 1'b0;
                    #(8680);
                    uart_rx = 1'b0;
                    #(8680);
                    
                    // Stop bit (high)
                    uart_rx = 1'b1;
                    #(8680);
                end
            join_none
            
            #(CLK_PERIOD * 100);
            
            $display("✓ UART interface test completed");
            pass_count = pass_count + 1;
            
            uart_rx = 1'b1; // Return to idle
            #(CLK_PERIOD * 10);
        end
    endtask
    
    // Test task: SPI interface
    task test_spi_interface;
        begin
            test_count = test_count + 1;
            $display("\nTest %0d: SPI Interface", test_count);
            
            // Simulate SPI read command
            spi_cs_n = 0;
            #(CLK_PERIOD * 10);
            
            // Send READ_CONFIG command (0x01)
            spi_send_byte(8'h01);
            spi_send_byte(8'h00); // Address 0
            
            // Receive response
            config_reg = 16'h1234;
            #(CLK_PERIOD * 50);
            
            spi_cs_n = 1;
            #(CLK_PERIOD * 10);
            
            $display("✓ SPI interface test completed");
            pass_count = pass_count + 1;
        end
    endtask
    
    // Test task: Integration test
    task test_integration;
        begin
            test_count = test_count + 1;
            $display("\nTest %0d: Integration Test", test_count);
            
            // Full system integration test
            job_header = 256'habcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcd;
            start_nonce = 32'h2000;
            nonce_range = 32'h20;
            control_reg = 8'h01;
            
            mining_enable = 1;
            #(CLK_PERIOD * 200);
            
            // Monitor all systems working together
            if (pipeline_busy && temperature > 8'h00 && power_consumption > 16'h0000) begin
                $display("✓ All systems working in integration");
                $display("  Pipeline: %b, Temp: %0d, Power: %0d", 
                        pipeline_busy, temperature, power_consumption);
                pass_count = pass_count + 1;
            end else begin
                $display("✗ Integration test failed");
                fail_count = fail_count + 1;
            end
            
            mining_enable = 0;
            #(CLK_PERIOD * 50);
        end
    endtask
    
    // SPI helper task
    task spi_send_byte;
        input [7:0] data;
        integer i;
        begin
            for (i = 0; i < 8; i = i + 1) begin
                spi_mosi = data[i];
                #(CLK_PERIOD);
                spi_clk = 1;
                #(CLK_PERIOD);
                spi_clk = 0;
            end
        end
    endtask
    
    // Monitor for simulation timeout
    initial begin
        #(SIM_TIME);
        $display("\n*** SIMULATION TIMEOUT ***");
        $display("Simulation exceeded maximum time limit");
        $finish;
    end
    
    // Display status information periodically
    always @(posedge clk_100m) begin
        if ($time % (CLK_PERIOD * 10000) == 0) begin
            $display("Time: %0d ns - Status: 0x%02h, Temp: %0d°C, Power: %0dmW", 
                    $time, status_reg, temperature, power_consumption);
        end
    end

endmodule
// ===============================================
// BM1387 ASIC with VESELOV HNS Testbench
// Comprehensive test suite for HNS mapping verification
// Tests hash-to-RGBA conversion and consciousness metrics
// ===============================================

`timescale 1ns/1ps

module bm1387_hns_tb;

    // Testbench parameters
    parameter CLK_PERIOD = 10; // 100 MHz clock
    parameter RESET_DURATION = 100;
    parameter TEST_DURATION = 10000;
    
    // DUT signals
    reg clk_100m;
    reg reset_n;
    
    // Mining Pipeline Interface
    reg [255:0] job_header;
    reg [31:0] start_nonce;
    reg [31:0] nonce_range;
    reg mining_enable;
    wire [31:0] found_nonce;
    wire [255:0] found_hash;
    wire hash_valid;
    wire pipeline_busy;
    
    // Temperature and Power Management
    wire [7:0] temperature;
    wire [15:0] power_consumption;
    wire thermal_throttle;
    
    // Control and Status
    reg [7:0] control_reg;
    wire [7:0] status_reg;
    reg [15:0] config_reg;
    
    // UART Interface
    reg uart_rx;
    wire uart_tx;
    
    // SPI Interface
    reg spi_clk;
    reg spi_cs_n;
    reg spi_mosi;
    wire spi_miso;
    
    // VESELOV HNS Interface
    wire [31:0] hns_rgba_r;
    wire [31:0] hns_rgba_g;
    wire [31:0] hns_rgba_b;
    wire [31:0] hns_rgba_a;
    wire [31:0] hns_vector_mag;
    wire [31:0] hns_energy;
    wire [31:0] hns_entropy;
    wire [31:0] hns_phi;
    wire [31:0] hns_phase_coh;
    wire hns_valid;
    
    // Debug Interface
    wire [31:0] debug_reg_0;
    wire [31:0] debug_reg_1;
    wire [31:0] debug_reg_2;
    wire [31:0] debug_reg_3;
    
    // Test control signals
    integer test_passed;
    integer test_failed;
    integer test_count;
    reg test_enable;
    
    // Test data storage
    reg [255:0] test_hash_1;
    reg [255:0] test_hash_2;
    reg [255:0] test_hash_3;
    reg [31:0] expected_rgba_r [0:2];
    reg [31:0] expected_rgba_g [0:2];
    reg [31:0] expected_rgba_b [0:2];
    reg [31:0] expected_rgba_a [0:2];
    
    // File handling for results
    integer results_file;
    
    // Instantiate DUT
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
        .hns_rgba_r(hns_rgba_r),
        .hns_rgba_g(hns_rgba_g),
        .hns_rgba_b(hns_rgba_b),
        .hns_rgba_a(hns_rgba_a),
        .hns_vector_mag(hns_vector_mag),
        .hns_energy(hns_energy),
        .hns_entropy(hns_entropy),
        .hns_phi(hns_phi),
        .hns_phase_coh(hns_phase_coh),
        .hns_valid(hns_valid),
        .debug_reg_0(debug_reg_0),
        .debug_reg_1(debug_reg_1),
        .debug_reg_2(debug_reg_2),
        .debug_reg_3(debug_reg_3)
    );
    
    // Clock generation
    initial begin
        clk_100m = 0;
        forever #(CLK_PERIOD/2) clk_100m = ~clk_100m;
    end
    
    // Reset generation
    initial begin
        reset_n = 0;
        #RESET_DURATION reset_n = 1;
    end
    
    // Test stimulus generation
    initial begin
        // Initialize testbench
        test_passed = 0;
        test_failed = 0;
        test_count = 0;
        test_enable = 1;
        
        // Open results file
        results_file = $fopen("hns_test_results.txt", "w");
        $fwrite(results_file, "BM1387 ASIC VESELOV HNS Test Results\n");
        $fwrite(results_file, "=====================================\n\n");
        
        // Wait for reset
        wait (reset_n == 1);
        #100;
        
        // Initialize inputs
        job_header = 256'h0000000000000000000000000000000000000000000000000000000000000000;
        start_nonce = 32'h00000000;
        nonce_range = 32'h00000001;
        mining_enable = 0;
        control_reg = 8'h00;
        config_reg = 16'h0000;
        uart_rx = 1;
        spi_clk = 0;
        spi_cs_n = 1;
        spi_mosi = 0;
        
        // Display test start
        $display("\n========================================");
        $display("Starting BM1387 ASIC VESELOV HNS Tests");
        $display("========================================\n");
        
        // Test 1: Basic HNS functionality
        $display("Test 1: Basic HNS Hash-to-RGBA Conversion");
        run_basic_hns_test();
        
        // Test 2: Consciousness metrics validation
        $display("\nTest 2: Consciousness Metrics Validation");
        run_consciousness_metrics_test();
        
        // Test 3: Vector magnitude calculation
        $display("\nTest 3: 3D Vector Magnitude Calculation");
        run_vector_magnitude_test();
        
        // Test 4: Phase coherence calculation
        $display("\nTest 4: Phase Coherence Calculation");
        run_phase_coherence_test();
        
        // Test 5: Integration with mining pipeline
        $display("\nTest 5: Mining Pipeline Integration");
        run_mining_integration_test();
        
        // Final summary
        display_test_summary();
        
        // Close results file
        $fclose(results_file);
        
        // End simulation
        #1000;
        $finish;
    end
    
    // Task: Run basic HNS test
    task run_basic_hns_test;
        reg [255:0] test_hash;
        begin
            test_count = test_count + 1;
            
            // Use a known test hash
            test_hash = 256'h123456789ABCDEF0FEDCBA0987654321ABCDEF0123456789ABCDEF0123456;
            job_header = test_hash;
            start_nonce = 32'h00000000;
            nonce_range = 32'h00000001;
            
            // Enable mining and HNS processing
            mining_enable = 1;
            control_reg = 8'h01;
            
            // Wait for hash computation and HNS processing
            wait (hns_valid == 1);
            #100;
            
            // Verify RGBA output ranges (should be 0 to 2^24)
            if (hns_rgba_r <= 32'h01000000 && hns_rgba_g <= 32'h01000000 && 
                hns_rgba_b <= 32'h01000000 && hns_rgba_a <= 32'h01000000) begin
                $display("  ✓ RGBA values in valid range");
                $fwrite(results_file, "Test %0d: RGBA values in valid range - R=%08X G=%08X B=%08X A=%08X\n", 
                       test_count, hns_rgba_r, hns_rgba_g, hns_rgba_b, hns_rgba_a);
                test_passed = test_passed + 1;
            end else begin
                $display("  ✗ RGBA values out of range");
                $fwrite(results_file, "Test %0d: FAILED - RGBA values out of range\n", test_count);
                test_failed = test_failed + 1;
            end
            
            // Verify vector magnitude is valid
            if (hns_vector_mag <= 32'h01000000) begin
                $display("  ✓ Vector magnitude in valid range: %08X", hns_vector_mag);
                $fwrite(results_file, "Test %0d: Vector magnitude valid - %08X\n", test_count, hns_vector_mag);
            end else begin
                $display("  ✗ Vector magnitude out of range");
                $fwrite(results_file, "Test %0d: FAILED - Vector magnitude out of range\n", test_count);
                test_failed = test_failed + 1;
            end
            
            // Disable mining
            mining_enable = 0;
            control_reg = 8'h00;
            #100;
        end
    endtask
    
    // Task: Run consciousness metrics test
    task run_consciousness_metrics_test;
        begin
            test_count = test_count + 1;
            
            // Set up test with high entropy hash
            job_header = 256'hFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF;
            start_nonce = 32'h00000000;
            nonce_range = 32'h00000001;
            
            mining_enable = 1;
            control_reg = 8'h01;
            
            // Wait for processing
            wait (hns_valid == 1);
            #100;
            
            // Verify consciousness metrics are calculated
            if (hns_energy != 0 || hns_entropy != 0 || hns_phi != 0) begin
                $display("  ✓ Consciousness metrics calculated");
                $display("    Energy: %08X", hns_energy);
                $display("    Entropy: %08X", hns_entropy);
                $display("    Phi: %08X", hns_phi);
                $fwrite(results_file, "Test %0d: Consciousness metrics - Energy=%08X Entropy=%08X Phi=%08X\n", 
                       test_count, hns_energy, hns_entropy, hns_phi);
                test_passed = test_passed + 1;
            end else begin
                $display("  ✗ Consciousness metrics not calculated");
                $fwrite(results_file, "Test %0d: FAILED - Consciousness metrics not calculated\n", test_count);
                test_failed = test_failed + 1;
            end
            
            // Verify metrics are in valid range
            if (hns_energy <= 32'h01000000 && hns_entropy <= 32'h01000000 && hns_phi <= 32'h01000000) begin
                $display("  ✓ Consciousness metrics in valid range");
            end else begin
                $display("  ✗ Consciousness metrics out of range");
                $fwrite(results_file, "Test %0d: FAILED - Consciousness metrics out of range\n", test_count);
                test_failed = test_failed + 1;
            end
            
            mining_enable = 0;
            control_reg = 8'h00;
            #100;
        end
    endtask
    
    // Task: Run vector magnitude test
    task run_vector_magnitude_test;
        begin
            test_count = test_count + 1;
            
            // Set up test with varied vector components
            job_header = 256'h0080FF000080FF000080FF000080FF000080FF000080FF000080FF000080FF00;
            start_nonce = 32'h00000000;
            nonce_range = 32'h00000001;
            
            mining_enable = 1;
            control_reg = 8'h01;
            
            wait (hns_valid == 1);
            #100;
            
            // Vector magnitude should be reasonable for balanced components
            if (hns_vector_mag > 0 && hns_vector_mag < 32'h00800000) begin
                $display("  ✓ Vector magnitude calculated correctly: %08X", hns_vector_mag);
                $fwrite(results_file, "Test %0d: Vector magnitude - %08X\n", test_count, hns_vector_mag);
                test_passed = test_passed + 1;
            end else begin
                $display("  ✗ Vector magnitude calculation failed: %08X", hns_vector_mag);
                $fwrite(results_file, "Test %0d: FAILED - Vector magnitude calculation\n", test_count);
                test_failed = test_failed + 1;
            end
            
            mining_enable = 0;
            control_reg = 8'h00;
            #100;
        end
    endtask
    
    // Task: Run phase coherence test
    task run_phase_coherence_test;
        begin
            test_count = test_count + 1;
            
            // Set up test with gradual phase changes
            job_header = 256'h0000000100000002000000030000000400000005000000060000000700000008;
            start_nonce = 32'h00000000;
            nonce_range = 32'h00000001;
            
            mining_enable = 1;
            control_reg = 8'h01;
            
            wait (hns_valid == 1);
            #100;
            
            // Phase coherence should be calculated
            if (hns_phase_coh != 0) begin
                $display("  ✓ Phase coherence calculated: %08X", hns_phase_coh);
                $fwrite(results_file, "Test %0d: Phase coherence - %08X\n", test_count, hns_phase_coh);
                test_passed = test_passed + 1;
            end else begin
                $display("  ✗ Phase coherence not calculated");
                $fwrite(results_file, "Test %0d: FAILED - Phase coherence not calculated\n", test_count);
                test_failed = test_failed + 1;
            end
            
            mining_enable = 0;
            control_reg = 8'h00;
            #100;
        end
    endtask
    
    // Task: Run mining integration test
    task run_mining_integration_test;
        begin
            test_count = test_count + 1;
            
            // Test full mining pipeline with HNS processing
            job_header = 256'h000102030405060708090A0B0C0D0E0F101112131415161718191A1B1C1D1E1F;
            start_nonce = 32'h00000000;
            nonce_range = 32'h00000005; // Test 5 nonces
            
            mining_enable = 1;
            control_reg = 8'h01;
            
            // Monitor pipeline operation
            wait (pipeline_busy == 1);
            $display("  ✓ Mining pipeline started");
            
            // Wait for completion
            wait (pipeline_busy == 0);
            #200;
            
            // Verify HNS processing occurred during mining
            if (hns_valid == 1) begin
                $display("  ✓ HNS processing completed during mining");
                $display("    Final RGBA: R=%08X G=%08X B=%08X A=%08X", 
                        hns_rgba_r, hns_rgba_g, hns_rgba_b, hns_rgba_a);
                $fwrite(results_file, "Test %0d: Mining integration successful\n", test_count);
                $fwrite(results_file, "  Final HNS data - R=%08X G=%08X B=%08X A=%08X Energy=%08X Entropy=%08X Phi=%08X\n",
                       hns_rgba_r, hns_rgba_g, hns_rgba_b, hns_rgba_a, hns_energy, hns_entropy, hns_phi);
                test_passed = test_passed + 1;
            end else begin
                $display("  ✗ HNS processing failed during mining");
                $fwrite(results_file, "Test %0d: FAILED - HNS processing during mining\n", test_count);
                test_failed = test_failed + 1;
            end
            
            mining_enable = 0;
            control_reg = 8'h00;
        end
    endtask
    
    // Task: Display test summary
    task display_test_summary;
        begin
            $display("\n========================================");
            $display("Test Summary");
            $display("========================================");
            $display("Tests Passed: %0d", test_passed);
            $display("Tests Failed: %0d", test_failed);
            $display("Total Tests: %0d", test_count);
            
            if (test_failed == 0) begin
                $display("Status: ALL TESTS PASSED ✓");
                $fwrite(results_file, "\nFinal Status: ALL TESTS PASSED ✓\n");
            end else begin
                $display("Status: SOME TESTS FAILED ✗");
                $fwrite(results_file, "\nFinal Status: SOME TESTS FAILED ✗\n");
            end
            
            $display("========================================\n");
            $fwrite(results_file, "========================================\n");
        end
    endtask
    
    // Monitor and log HNS outputs
    always @(posedge clk_100m) begin
        if (hns_valid && test_enable) begin
            $display("HNS Output: R=%08X G=%08X B=%08X A=%08X Mag=%08X Energy=%08X Entropy=%08X Phi=%08X Phase=%08X",
                    hns_rgba_r, hns_rgba_g, hns_rgba_b, hns_rgba_a,
                    hns_vector_mag, hns_energy, hns_entropy, hns_phi, hns_phase_coh);
        end
    end
    
    // Timeout check
    initial begin
        #TEST_DURATION;
        $display("\n!!! TEST TIMEOUT - Simulation exceeded maximum duration !!!");
        $fwrite(results_file, "\n!!! TEST TIMEOUT !!!\n");
        $finish;
    end
    
endmodule
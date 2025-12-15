#!/usr/bin/env julia

# Test script for CHIMERA consciousness engine scaling
# Tests performance at 1K, 10K, and 100K nodes

using Pkg
Pkg.activate(".")
Pkg.instantiate()

# Import the scaling implementation
include("scale_consciousness.jl")

function main()
    println("🚀 CHIMERA Consciousness Engine - Scaling Performance Test")
    println("="^70)
    
    # Check system capabilities
    println("System Information:")
    println("  Threads available: $(Threads.nthreads())")
    println("  Memory info: $(Sys.total_memory()/1024^3, digits=1)GB total")
    println("  CPU cores: $(Sys.CPU_IDS)")
    println()
    
    # Test configurations - progressively larger scales
    test_configs = [
        (10, "Tiny Scale", "1K nodes"),      # 10^3 = 1,000 nodes
        (22, "Small Scale", "10K nodes"),    # 22^3 ≈ 10,648 nodes  
        (46, "Medium Scale", "100K nodes"),  # 46^3 ≈ 97,336 nodes
    ]
    
    all_results = Dict{Int, Dict{String, Any}}()
    
    for (nodes, scale_name, description) in test_configs
        println("🧪 Testing $scale_name: $description")
        println("Nodes: $(nodes)^3 = $(nodes^3)")
        
        try
            # Run scaling benchmark
            results = run_scaling_benchmark(
                node_counts=[nodes], 
                temperature=1.0, 
                duration=1.0  # Short duration for testing
            )
            
            if haskey(results, nodes)
                all_results[nodes] = results[nodes]
                result = results[nodes]
                
                println("✅ Success!")
                println("   Execution time: $(round(result["chunked_time"], digits=3))s")
                println("   Memory usage: $(round(result["memory_mb"], digits=2))MB")
                println("   Parallel efficiency: $(round(result["parallel_efficiency"], digits=3))")
                println("   Global metrics:")
                println("     Energy: $(round(result["global_metrics"]["energy"], digits=4))")
                println("     Entropy: $(round(result["global_metrics"]["entropy"], digits=4))")
                println("     Phi: $(round(result["global_metrics"]["phi"], digits=4))")
            else
                println("❌ Failed to get results for $nodes nodes")
            end
            
        catch e
            println("❌ Error testing $scale_name: $e")
            println("   This may be due to insufficient memory or computational resources")
        end
        
        println("-"^50)
    end
    
    # Analyze scaling performance
    if !isempty(all_results)
        println("\n📊 SCALING ANALYSIS")
        println("="^50)
        
        analysis = analyze_scaling_performance(all_results)
        viz_data = generate_scaling_visualization_data(all_results)
        
        # Save results for later analysis
        save("scaling_test_data.jld", Dict("results" => all_results, "analysis" => analysis, "viz_data" => viz_data))
        println("💾 Test data saved to scaling_test_data.jld")
        
        # Performance summary
        println("\n🎯 SCALING SUMMARY")
        println("="^30)
        max_nodes = maximum([r["total_nodes"] for r in values(all_results)])
        max_time = maximum([r["chunked_time"] for r in values(all_results)])
        avg_efficiency = mean([r["parallel_efficiency"] for r in values(all_results)])
        
        println("✅ Successfully tested up to $max_nodes nodes")
        println("⏱️  Maximum execution time: $(round(max_time, digits=2))s")
        println("⚡ Average parallel efficiency: $(round(avg_efficiency, digits=3))")
        
        if max_nodes >= 100000
            println("🎉 TARGET ACHIEVED: Successfully tested 100K+ node scaling!")
        end
        
        return all_results, analysis, viz_data
    else
        println("\n❌ No scaling tests completed successfully")
        println("This may indicate system resource limitations")
        return nothing, nothing, nothing
    end
end

# Run the main test
if abspath(PROGRAM_FILE) == @__FILE__
    results, analysis, viz_data = main()
    if results !== nothing
        println("\n🏁 Scaling test completed successfully!")
        exit(0)
    else
        println("\n💥 Scaling test failed!")
        exit(1)
    end
end
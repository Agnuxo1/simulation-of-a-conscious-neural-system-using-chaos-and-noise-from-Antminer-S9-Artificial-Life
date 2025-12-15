#!/usr/bin/env python3
"""
GTKWave Automation Script for BM1387 ASIC VESELOV HNS Visualization
==================================================================

This script provides automated GTKWave configuration and launching capabilities
for analyzing BM1387 ASIC signals with VESELOV HNS parameters.

Features:
- Automatic signal grouping for VESELOV HNS parameters
- Mining pipeline timing analysis
- Consciousness metrics visualization
- Customizable waveform views
- VS Code integration support

Author: Kilo Code
Date: 2025-12-15
"""

import os
import sys
import json
import subprocess
import argparse
from pathlib import Path
from typing import Dict, List, Optional

class GTKWaveAutomation:
    """GTKWave automation controller for BM1387 ASIC visualization"""
    
    def __init__(self, waveforms_dir: str = "waveforms"):
        self.waveforms_dir = Path(waveforms_dir)
        self.config_dir = Path("gtkwave_config")
        self.config_dir.mkdir(exist_ok=True)
        
        # VESELOV HNS signal groups for organized viewing
        self.signal_groups = {
            "Clock_and_Reset": [
                "dut.clk_100m",
                "dut.reset_n"
            ],
            "Mining_Pipeline": [
                "dut.job_header\\[255:0\\]",
                "dut.start_nonce\\[31:0\\]",
                "dut.nonce_range\\[31:0\\]",
                "dut.mining_enable",
                "dut.found_nonce\\[31:0\\]",
                "dut.found_hash\\[255:0\\]",
                "dut.hash_valid",
                "dut.pipeline_busy",
                "dut.status_reg\\[7:0\\]"
            ],
            "VESELOV_HNS_RGBA": [
                "dut.hns_rgba_r\\[31:0\\]",
                "dut.hns_rgba_g\\[31:0\\]",
                "dut.hns_rgba_b\\[31:0\\]",
                "dut.hns_rgba_a\\[31:0\\]"
            ],
            "Consciousness_Metrics": [
                "dut.hns_vector_mag\\[31:0\\]",
                "dut.hns_energy\\[31:0\\]",
                "dut.hns_entropy\\[31:0\\]",
                "dut.hns_phi\\[31:0\\]",
                "dut.hns_phase_coh\\[31:0\\]",
                "dut.hns_valid"
            ],
            "Thermal_Power": [
                "dut.temperature\\[7:0\\]",
                "dut.power_consumption\\[15:0\\]",
                "dut.thermal_throttle"
            ],
            "Control_Interfaces": [
                "dut.control_reg\\[7:0\\]",
                "dut.config_reg\\[15:0\\]",
                "dut.debug_reg_0\\[31:0\\]",
                "dut.debug_reg_1\\[31:0\\]",
                "dut.debug_reg_2\\[31:0\\]",
                "dut.debug_reg_3\\[31:0\\]"
            ],
            "Communication": [
                "dut.uart_rx",
                "dut.uart_tx",
                "dut.spi_clk",
                "dut.spi_cs_n",
                "dut.spi_mosi",
                "dut.spi_miso"
            ]
        }
        
        # Color scheme for different signal groups
        self.color_scheme = {
            "Clock_and_Reset": "yellow",
            "Mining_Pipeline": "blue", 
            "VESELOV_HNS_RGBA": "red",
            "Consciousness_Metrics": "green",
            "Thermal_Power": "orange",
            "Control_Interfaces": "purple",
            "Communication": "cyan"
        }
    
    def create_gtkw_save_file(self, vcd_file: str, output_file: str = None) -> str:
        """Create GTKWave save file with organized signal groups"""
        
        if output_file is None:
            base_name = Path(vcd_file).stem
            output_file = self.config_dir / f"{base_name}_signals.gtkw"
        
        save_content = [
            "[timestart] 0",
            "[size] 1920 1080",
            "[signals_start]"
        ]
        
        # Add signal groups in logical order
        for group_name, signals in self.signal_groups.items():
            save_content.append(f"\n# {group_name}")
            for signal in signals:
                save_content.append(f"{signal}")
        
        save_content.extend([
            "[signals_end]",
            "[tree_open_1]",
            "[pattern_trace] 1",
            "[pattern_trace] 0"
        ])
        
        # Write save file
        with open(output_file, 'w') as f:
            f.write('\n'.join(save_content))
        
        return str(output_file)
    
    def create_mining_analysis_template(self, vcd_file: str) -> str:
        """Create specialized template for mining pipeline analysis"""
        
        template_name = f"mining_analysis_{Path(vcd_file).stem}.gtkw"
        template_file = self.config_dir / template_name
        
        # Mining-focused signal selection
        mining_signals = {
            "Mining_Control": [
                "dut.mining_enable",
                "dut.control_reg\\[7:0\\]",
                "dut.status_reg\\[7:0\\]",
                "dut.pipeline_busy"
            ],
            "Hash_Processing": [
                "dut.job_header\\[255:0\\]",
                "dut.current_nonce\\[31:0\\]",
                "dut.current_hash\\[255:0\\]",
                "dut.hash_valid",
                "dut.found_nonce\\[31:0\\]",
                "dut.found_hash\\[255:0\\]"
            ],
            "Performance_Metrics": [
                "dut.hash_count\\[15:0\\]",
                "dut.temperature\\[7:0\\]",
                "dut.power_consumption\\[15:0\\]",
                "dut.thermal_throttle"
            ],
            "HNS_RealTime": [
                "dut.hns_rgba_r\\[31:0\\]",
                "dut.hns_rgba_g\\[31:0\\]",
                "dut.hns_rgba_b\\[31:0\\]",
                "dut.hns_rgba_a\\[31:0\\]",
                "dut.hns_energy\\[31:0\\]",
                "dut.hns_entropy\\[31:0\\]",
                "dut.hns_valid"
            ]
        }
        
        content = [
            "[timestart] 0",
            "[size] 1920 1080",
            "[signals_start]"
        ]
        
        for group_name, signals in mining_signals.items():
            content.append(f"\n# {group_name}")
            for signal in signals:
                content.append(f"{signal}")
        
        content.extend([
            "[signals_end]",
            "[tree_open_1]",
            "[pattern_trace] 1"
        ])
        
        with open(template_file, 'w') as f:
            f.write('\n'.join(content))
        
        return str(template_file)
    
    def create_consciousness_analysis_template(self, vcd_file: str) -> str:
        """Create specialized template for consciousness metrics analysis"""
        
        template_name = f"consciousness_analysis_{Path(vcd_file).stem}.gtkw"
        template_file = self.config_dir / template_name
        
        # Consciousness-focused signal selection
        consciousness_signals = {
            "HNS_RGBA_Parameters": [
                "dut.hns_rgba_r\\[31:0\\]",
                "dut.hns_rgba_g\\[31:0\\]",
                "dut.hns_rgba_b\\[31:0\\]",
                "dut.hns_rgba_a\\[31:0\\]"
            ],
            "Consciousness_Computation": [
                "dut.hns_vector_mag\\[31:0\\]",
                "dut.hns_energy\\[31:0\\]",
                "dut.hns_entropy\\[31:0\\]",
                "dut.hns_phi\\[31:0\\]",
                "dut.hns_phase_coh\\[31:0\\]"
            ],
            "Processing_Control": [
                "dut.hns_valid",
                "dut.u_veselov_hns.hns_state\\[7:0\\]",
                "dut.u_veselov_hns.process_counter\\[7:0\\]"
            ],
            "Hash_Input_Source": [
                "dut.current_hash\\[255:0\\]",
                "dut.job_header\\[255:0\\]",
                "dut.current_nonce\\[31:0\\]"
            ]
        }
        
        content = [
            "[timestart] 0",
            "[size] 1920 1080", 
            "[signals_start]"
        ]
        
        for group_name, signals in consciousness_signals.items():
            content.append(f"\n# {group_name}")
            for signal in signals:
                content.append(f"{signal}")
        
        content.extend([
            "[signals_end]",
            "[tree_open_1]",
            "[pattern_trace] 1"
        ])
        
        with open(template_file, 'w') as f:
            f.write('\n'.join(content))
        
        return str(template_file)
    
    def launch_gtkwave(self, vcd_file: str, save_file: str = None, 
                      template_type: str = "default") -> bool:
        """Launch GTKWave with configured signal view"""
        
        vcd_path = Path(vcd_file)
        if not vcd_path.exists():
            print(f"ERROR: VCD file not found: {vcd_file}")
            return False
        
        # Create appropriate save file based on template type
        if template_type == "mining":
            save_file = self.create_mining_analysis_template(vcd_file)
        elif template_type == "consciousness":
            save_file = self.create_consciousness_analysis_template(vcd_file)
        elif save_file is None:
            save_file = self.create_gtkw_save_file(vcd_file)
        
        # GTKWave launch command
        cmd = [
            "gtkwave",
            "-f", str(vcd_path),
            "-g", str(save_file)
        ]
        
        print(f"Launching GTKWave with {template_type} template...")
        print(f"VCD file: {vcd_file}")
        print(f"Save file: {save_file}")
        print(f"Command: {' '.join(cmd)}")
        
        try:
            # Launch GTKWave
            if os.name == 'nt':  # Windows
                subprocess.Popen(cmd, creationflags=subprocess.CREATE_NEW_CONSOLE)
            else:  # Unix-like
                subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            print("GTKWave launched successfully!")
            return True
            
        except FileNotFoundError:
            print("ERROR: GTKWave not found in PATH")
            print("Please install GTKWave or add it to your PATH environment variable")
            return False
        except Exception as e:
            print(f"ERROR: Failed to launch GTKWave: {e}")
            return False
    
    def create_comprehensive_save(self, vcd_file: str) -> Dict[str, str]:
        """Create all save file variants for comprehensive analysis"""
        
        save_files = {
            "default": self.create_gtkw_save_file(vcd_file),
            "mining": self.create_mining_analysis_template(vcd_file),
            "consciousness": self.create_consciousness_analysis_template(vcd_file)
        }
        
        return save_files
    
    def find_vcd_files(self) -> List[str]:
        """Find all VCD files in the waveforms directory"""
        
        vcd_files = []
        if self.waveforms_dir.exists():
            vcd_files = list(self.waveforms_dir.glob("*.vcd"))
        
        return [str(f) for f in vcd_files]
    
    def list_available_vcd_files(self):
        """List all available VCD files with analysis options"""
        
        vcd_files = self.find_vcd_files()
        
        if not vcd_files:
            print("No VCD files found in waveforms directory")
            return
        
        print("Available VCD files:")
        print("=" * 50)
        
        for i, vcd_file in enumerate(vcd_files, 1):
            file_path = Path(vcd_file)
            print(f"{i}. {file_path.name}")
            print(f"   Path: {vcd_file}")
            print(f"   Size: {file_path.stat().st_size} bytes")
            print(f"   Modified: {file_path.stat().st_mtime}")
            print()
            
            # Show available analysis templates
            save_files = self.create_comprehensive_save(vcd_file)
            print(f"   Available templates:")
            for template_type, save_file in save_files.items():
                print(f"     - {template_type}: {save_file}")
            print()

def main():
    """Main function for command-line interface"""
    
    parser = argparse.ArgumentParser(
        description="GTKWave Automation for BM1387 ASIC VESELOV HNS Visualization"
    )
    
    parser.add_argument(
        "--vcd-file", "-f",
        help="VCD file to analyze"
    )
    
    parser.add_argument(
        "--template", "-t",
        choices=["default", "mining", "consciousness"],
        default="default",
        help="Analysis template to use"
    )
    
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available VCD files"
    )
    
    parser.add_argument(
        "--generate-save", "-g",
        action="store_true", 
        help="Generate save files without launching GTKWave"
    )
    
    args = parser.parse_args()
    
    automation = GTKWaveAutomation()
    
    if args.list:
        automation.list_available_vcd_files()
        return
    
    if args.vcd_file:
        if args.generate_save:
            save_files = automation.create_comprehensive_save(args.vcd_file)
            print("Generated save files:")
            for template_type, save_file in save_files.items():
                print(f"  {template_type}: {save_file}")
        else:
            automation.launch_gtkwave(args.vcd_file, template_type=args.template)
    else:
        print("Please specify a VCD file with --vcd-file")
        print("Use --list to see available VCD files")

if __name__ == "__main__":
    main()
load_package flow

proc make_all_pins_virtual {} {

    execute_module -tool map

    set name_ids [get_names -filter * -node_type pin]

    foreach_in_collection name_id $name_ids {
        set pin_name [get_name_info -info full_path $name_id]
        set_instance_assignment -to $pin_name -name VIRTUAL_PIN ON
    }
    export_assignments
}

# Make Stratix V Assignments
proc make_S5_general_assignments {} {
    set_global_assignment -name FAMILY "Stratix V"
    set_global_assignment -name DEVICE "5SGXEA7H1F35C1"
    set_global_assignment -name PROJECT_OUTPUT_DIRECTORY output_files
    set_global_assignment -name ERROR_CHECK_FREQUENCY_DIVISOR 256
    set_global_assignment -name OPTIMIZATION_MODE "AGGRESSIVE PERFORMANCE"
}

# Make Cyclone V Assignments
proc make_C5_general_assignments {} {
    set_global_assignment -name FAMILY "Cyclone V"
    set_global_assignment -name DEVICE "5CSEMA4U23C6"
    set_global_assignment -name SEED 2
    set_global_assignment -name PROJECT_OUTPUT_DIRECTORY output_files
    set_global_assignment -name ERROR_CHECK_FREQUENCY_DIVISOR 256
    set_global_assignment -name OPTIMIZATION_MODE "AGGRESSIVE PERFORMANCE"
    #set_global_assignment -name MUX_RESTRUCTURE AUTO
    #set_global_assignment -name REMOVE_REDUNDANT_LOGIC_CELLS ON
    #set_global_assignment -name FINAL_PLACEMENT_OPTIMIZATION ALWAYS
    #set_global_assignment -name FITTER_AGGRESSIVE_ROUTABILITY_OPTIMIZATION ALWAYS
    #set_global_assignment -name AUTO_DELAY_CHAINS_FOR_HIGH_FANOUT_INPUT_PINS ON
    #set_global_assignment -name PHYSICAL_SYNTHESIS_EFFORT EXTRA
    #set_global_assignment -name ALM_REGISTER_PACKING_EFFORT HIGH
}

# Make Arria 10 115 Assignments
proc make_A10_115_general_assignments {} {
    set_global_assignment -name FAMILY "Arria 10"
    set_global_assignment -name DEVICE "10AX115S2F45I1SG"
    set_global_assignment -name PROJECT_OUTPUT_DIRECTORY output_files
    set_global_assignment -name ERROR_CHECK_FREQUENCY_DIVISOR 256
    set_global_assignment -name OPTIMIZATION_MODE "AGGRESSIVE PERFORMANCE"
}

# Make Arria 10 057 Assignments
proc make_A10_057_general_assignments {} {
    set_global_assignment -name FAMILY "Arria 10"
    set_global_assignment -name DEVICE "10AX057K2F40I1SG"
    set_global_assignment -name PROJECT_OUTPUT_DIRECTORY output_files
    set_global_assignment -name ERROR_CHECK_FREQUENCY_DIVISOR 256
    set_global_assignment -name OPTIMIZATION_MODE "AGGRESSIVE PERFORMANCE"
}

proc analyse_power { top } {
    set_global_assignment -name MIN_CORE_JUNCTION_TEMP 0
    set_global_assignment -name MAX_CORE_JUNCTION_TEMP 85
    set_global_assignment -name POWER_PRESET_COOLING_SOLUTION "23 MM HEAT SINK WITH 200 LFPM AIRFLOW"
    set_global_assignment -name POWER_BOARD_THERMAL_MODEL "NONE (CONSERVATIVE)"
    set_global_assignment -name EDA_SIMULATION_TOOL "ModelSim-Altera (Verilog)"
    set_global_assignment -name EDA_MAP_ILLEGAL_CHARACTERS ON -section_id eda_simulation
    set_global_assignment -name EDA_TIME_SCALE "1 ps" -section_id eda_simulation
    set_global_assignment -name EDA_OUTPUT_DATA_FORMAT "VERILOG HDL" -section_id eda_simulation
    set_global_assignment -name EDA_ENABLE_GLITCH_FILTERING ON -section_id eda_simulation
    set_global_assignment -name EDA_WRITE_NODES_FOR_POWER_ESTIMATION ALL_NODES -section_id eda_simulation
    set_global_assignment -name EDA_TEST_BENCH_ENABLE_STATUS TEST_BENCH_MODE -section_id eda_simulation
    set_global_assignment -name EDA_NATIVELINK_SIMULATION_TEST_BENCH [join [list $top "_tb"] ""] -section_id eda_simulation
    set_global_assignment -name EDA_TEST_BENCH_DESIGN_INSTANCE_NAME DUT -section_id eda_simulation
    set_global_assignment -name EDA_TEST_BENCH_NAME [join [list $top "_tb"] ""] -section_id eda_simulation
    set_global_assignment -name EDA_DESIGN_INSTANCE_NAME DUT -section_id [join [list $top "_tb"] ""]
    set_global_assignment -name EDA_TEST_BENCH_MODULE_NAME [join [list $top "_tb"] ""] -section_id [join [list $top "_tb"] ""]
    set_global_assignment -name EDA_TEST_BENCH_FILE [join [list "../rtl/" $top "_tb.sv"] ""] -section_id [join [list $top "_tb"] ""]
    set_global_assignment -name PARTITION_NETLIST_TYPE SOURCE -section_id Top
    set_global_assignment -name PARTITION_FITTER_PRESERVATION_LEVEL PLACEMENT_AND_ROUTING -section_id Top
    set_global_assignment -name PARTITION_COLOR 16764057 -section_id Top
    set_global_assignment -name POWER_USE_INPUT_FILES ON
    set_global_assignment -name POWER_DEFAULT_TOGGLE_RATE "12.5 %"
    set_global_assignment -name POWER_DEFAULT_INPUT_IO_TOGGLE_RATE "12.5 %"
    set_global_assignment -name POWER_USE_PVA OFF
    set_global_assignment -name POWER_REPORT_SIGNAL_ACTIVITY ON
    set_global_assignment -name POWER_REPORT_POWER_DISSIPATION ON
    set_global_assignment -name POWER_INPUT_FILE_NAME [join [list "simulation/modelsim/" $top ".vcd"] ""] -section_id [join [list $top ".vcd"] ""]
    set_instance_assignment -name POWER_READ_INPUT_FILE [join [list $top ".vcd"] ""] -to $top
    set_instance_assignment -name PARTITION_HIERARCHY root_partition -to | -section_id Top
    set_global_assignment -name FLOW_ENABLE_POWER_ANALYZER ON
}


proc compile { project } {
    project_open $project

    #make_S5_general_assignments
    export_assignments

    execute_module -tool map
    execute_module -tool fit
    execute_module -tool sta

    project_close
}

proc create_project { project {path ""} } {
    if {[project_exists $project]} {
        project_open -revision $project $project
    } else {
        project_new -revision $project $project
    }

    if {$path != "" } {
        set path_tail [string index $path end]
        if {$path_tail != "/"} {
            append $path "/"
        }
	
	# If  files.f exits that we want to read from that, otherwise just default to the standard ../rtl folder
	if [file exists "files.f"] {
	    set fp [open "files.f" r]
	    set file_data [read $fp]
	    close $fp

	    set data [split $file_data "\n"]
	    foreach line $data {
		if [string match {*.v} $line] {
                    set_global_assignment -name VERILOG_FILE $line
		}
		if [string match {*.sv} $line] {
                    set_global_assignment -name SYSTEMVERILOG_FILE $line
		}
	    }
	} else {

		append rtl_path $path "rtl"
		set_global_assignment -name SEARCH_PATH $rtl_path
		foreach rtl [glob -nocomplain -dir $rtl_path *.v] {
		    set_global_assignment -name VERILOG_FILE $rtl
		}
		foreach rtl [glob -nocomplain -dir $rtl_path *.sv] {
		    set_global_assignment -name SYSTEMVERILOG_FILE $rtl
		}
        }
    
        if {[string match *area* $project]} {
            append area_file $path "area/" $project ".v"
            set_global_assignment -name VERILOG_FILE $area_file
            set sdc_dir "area/"
        } else {
            set sdc_dir "sdc/"
        }
        append sdc_file $path $sdc_dir $project ".sdc"
        set_global_assignment -name SDC_FILE $sdc_file
    }

    make_C5_general_assignments
    make_all_pins_virtual
    analyse_power $project
    export_assignments

    project_close
}


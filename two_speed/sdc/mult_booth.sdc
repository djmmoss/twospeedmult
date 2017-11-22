#create_clock -period 1.9 [get_ports clk]
set_max_delay -from {mult_booth_control:INST_MULT_BOOTH_CONTROL|multipland*} -through [get_pins {INST_MULT_BOOTH_DATA|*|*}] -to {mult_booth_control:INST_MULT_BOOTH_CONTROL|p*} 3.60
set_max_delay -from {mult_booth_control:INST_MULT_BOOTH_CONTROL|p*} -through [get_pins {INST_MULT_BOOTH_DATA|*|*}] -to {mult_booth_control:INST_MULT_BOOTH_CONTROL|p*} 3.6
set_max_delay -from {mult_booth_control:INST_MULT_BOOTH_CONTROL|p_d*} -through [get_pins {INST_MULT_BOOTH_DATA|*|*}] -to {mult_booth_control:INST_MULT_BOOTH_CONTROL|p*} 3.6

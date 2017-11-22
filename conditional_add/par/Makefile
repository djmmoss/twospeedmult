PROJ=mult_booth
DPATH=../
SRC=$(DPATH)rtl/*.sv
TCL=./*.tcl

.PHONY: all
all: createproject power

.PHONY: createproject
createproject: $(DPATH) $(TCL)
	quartus_sh --tcl_eval source ./tcl_command_pool.tcl\; create_project $(PROJ) $(DPATH)

.PHONY: compile
compile:
	quartus_sh --tcl_eval source ./tcl_command_pool.tcl\; compile $(PROJ)

.PHONY: power
power: sim
	quartus_pow --read_settings_files=off --write_settings_files=off $(PROJ) -c $(PROJ)

.PHONY: eda
eda: compile
	quartus_eda --read_settings_files=off --write_settings_files=off $(PROJ) -c $(PROJ)

.PHONY: sim
sim: eda
	quartus_sh -t /opt/intelFPGA/17.0/quartus/common/tcl/internal/nativelink/qnativesim.tcl --no_gui $(PROJ) $(PROJ)


.PHONY: clean
clean:
	rm -rf *.qsf *.qpf *.qws *rpt *.txt *db output_files simulation &> /dev/null

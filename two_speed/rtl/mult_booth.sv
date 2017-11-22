module mult_booth #(DATA_WIDTH = 32) (
	input  logic                    clk    ,
	input  logic [  DATA_WIDTH-1:0] i_a    ,
	input  logic [  DATA_WIDTH-1:0] i_b    ,
	input  logic                    i_valid,
	output logic                    o_valid,
	output logic [2*DATA_WIDTH-1:0] o_c
);

	logic [DATA_WIDTH-1:0] multipland = 0;
	logic [DATA_WIDTH-1:0] add           ;
	logic [DATA_WIDTH-1:0] shift         ;
	logic [           2:0] p_d           ;

    mult_booth_data #(DATA_WIDTH) INST_MULT_BOOTH_DATA (
        .data_p_d        (p_d),
        .data_multipland (multipland),
        .data_shift      (shift),
        .data_add        (add)
    );

	// Critial Path
	mult_booth_control #(DATA_WIDTH) INST_MULT_BOOTH_CONTROL (
		.clk       (clk       ),
		.i_a       (i_a       ),
		.i_b       (i_b       ),
		.i_valid   (i_valid   ),
		.add       (add       ),
		.multipland(multipland),
		.shifted   (shift     ),
		.p_d       (p_d       ),
		.o_valid   (o_valid   ),
		.o_c       (o_c       )
	);

endmodule

module mult_booth_data # (DATA_WIDTH = 32) (
    input  logic [2:0]            data_p_d,
    input  logic [DATA_WIDTH-1:0] data_multipland,
    input  logic [DATA_WIDTH-1:0] data_shift,
    output logic [DATA_WIDTH-1:0] data_add
);
	
    logic [DATA_WIDTH-1:0] pp            ;
	logic                  d, s, a, c;

	// Do the Booth Encoder
	booth2encoder_nz INST_BOOTH_ENCODE (data_p_d[0],data_p_d[1],data_p_d[2],d,a);

	// Generate the Partial Product
	booth2ppg_nz #(DATA_WIDTH) INST_BOOTH_PPG (data_multipland,d,a,pp);

	// Shift the Results
	adder #(DATA_WIDTH) INST_ADDER (data_shift,pp+data_p_d[2],data_add);

    endmodule

module mult_booth_control #(DATA_WIDTH = 32) (
	input  logic                    clk       ,
	input  logic [  DATA_WIDTH-1:0] i_a       ,
	input  logic [  DATA_WIDTH-1:0] i_b       ,
	input  logic                    i_valid   ,
	input  logic [  DATA_WIDTH-1:0] add       ,
	output logic [  DATA_WIDTH-1:0] multipland,
	output logic [  DATA_WIDTH-1:0] shifted   ,
	output logic [             2:0] p_d       ,
	output logic                    o_valid   ,
	output logic [2*DATA_WIDTH-1:0] o_c
);

	logic [          2*DATA_WIDTH-1:0] p     = 0;
	logic [          2*DATA_WIDTH-1:0] shift    ;
	logic [$clog2(DATA_WIDTH/2 + 1):0] count = 0;
	logic                              delay    ;
	logic [          DATA_WIDTH-1:0] res      ;
	logic                              s        ;
	logic flag;

	shifter #(2*DATA_WIDTH,2) INST_SHIFTER (p,shift);

	assign shifted = shift[2*DATA_WIDTH-1:DATA_WIDTH];
	assign res = s ? add : shifted;

	always_ff @(posedge clk) begin
		unique casez ({i_valid, delay})
			2'b1z: begin
				p          <= {{DATA_WIDTH{1'b0}}, i_b};
				p_d        <= {i_b[1:0], 1'b0};
                s          <= ~(&{i_b[1:0], 1'b0} | &(~{i_b[1:0], 1'b0}));
				multipland <= i_a;
				count      <= 0;
				delay      <= (&{i_b[1:0], 1'b0} | &(~{i_b[1:0], 1'b0}));
			end
			2'b01: begin
				p_d   <= p[3:1];
                s     <= ~(&p[3:1] | &(~p[3:1]));
				p     <= {res, p[DATA_WIDTH+1:2]};
				delay <= (&p[3:1] | &(~p[3:1]));
				count <= count + 1;
			end
			2'b00: begin
				delay <= 1'b1;
			end
		endcase
	end

	assign o_c     = p;
	assign o_valid = (count == (DATA_WIDTH/2 + 1));

endmodule

module adder #(DATA_WIDTH = 32) (
	input  logic [DATA_WIDTH-1:0] a,
	input  logic [DATA_WIDTH-1:0] b,
	output logic [DATA_WIDTH-1:0] c
);

	assign c = a + b;

endmodule

module adder_dual #(DATA_WIDTH = 32) (
	input  logic [DATA_WIDTH-1:0] a,
	input  logic [DATA_WIDTH-1:0] b,
	input  logic [DATA_WIDTH-1:0] c,
	output logic [DATA_WIDTH-1:0] d
);

	assign d = a + b + c;

endmodule

module shifter #(DATA_WIDTH = 32, N = 2) (
	input  logic [DATA_WIDTH-1:0] a,
	output logic [DATA_WIDTH-1:0] b
);

	assign b = {{N{a[DATA_WIDTH-1]}}, a[DATA_WIDTH-1:N]};

endmodule

module booth2encoder_nz (
	input  logic a1,
	input  logic a2,
	input  logic a3,
	output logic d ,
	output logic a
);

	assign d = a3;
	assign a = a2 ^ a1;

endmodule // booth2encoder

module booth2ppg_nz #(parameter DATA_WIDTH = 32) (
	input  logic [DATA_WIDTH-1:0] x,
	input  logic                  d,
	input  logic                  a,
	output logic [DATA_WIDTH-1:0] p
);

	genvar i;

	logic [DATA_WIDTH-1:0] neg_x;

	logic [DATA_WIDTH-1:0] d_res;

	assign d_res[0] = d ? ~x[0] : x[0];
	assign p[0]     = a ? d_res[0] : d;

	generate
		for (i = 1; i < DATA_WIDTH; i++) begin : GEN_MUX_PPG
			assign d_res[i] = d ? ~x[i] : x[i];
			assign p[i] = a ? d_res[i] : d_res[i-1];
		end
	endgenerate

endmodule // booth2ppg

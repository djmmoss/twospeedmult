module mult #(DATA_WIDTH = 32) (
	input  logic                    clk    ,
	input  logic [  DATA_WIDTH-1:0] i_a    ,
	input  logic [  DATA_WIDTH-1:0] i_b    ,
	output logic [2*DATA_WIDTH-1:0] o_c
);

    logic [DATA_WIDTH-1:0] a, b;
    logic [2*DATA_WIDTH-1:0] c;

    lpm # (DATA_WIDTH) (
        .clk    (clk),
        .dataa  (a),
        .datab  (b),
        .result (c)
    );

    always_ff @(posedge clk) begin
        a <= i_a;
        b <= i_b;
        o_c <= c;
    end

endmodule

module lpm # (DATA_WIDTH = 32) (
    input  logic                    clk,
	input  logic [DATA_WIDTH-1:0]   dataa,
	input  logic [DATA_WIDTH-1:0]   datab,
	output logic [2*DATA_WIDTH-1:0] result
);


	lpm_mult	lpm_mult_component (
				.dataa (dataa),
				.datab (datab),
				.result (result),
				.aclr (1'b0),
				.clken (1'b1),
//				.clock (clk),
				.sclr (1'b0),
				.sum (1'b0));
	defparam
		lpm_mult_component.lpm_hint = "MAXIMIZE_SPEED=9, DEDICATED_MULTIPLIER_CIRCUITRY=NO",
        lpm_mult_component.lpm_pipeline = 0,
		lpm_mult_component.lpm_representation = "SIGNED",
		lpm_mult_component.lpm_type = "LPM_MULT",
		lpm_mult_component.lpm_widtha = DATA_WIDTH,
		lpm_mult_component.lpm_widthb = DATA_WIDTH,
		lpm_mult_component.lpm_widthp = 2*DATA_WIDTH;


endmodule

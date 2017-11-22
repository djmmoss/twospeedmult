module mult #(DATA_WIDTH = 32) (
	input  logic                    clk    ,
	input  logic [  DATA_WIDTH-1:0] i_a    ,
	input  logic [  DATA_WIDTH-1:0] i_b    ,
	input  logic                    i_valid,
	output logic                    o_valid,
	output logic [2*DATA_WIDTH-1:0] o_c
);

	logic [DATA_WIDTH-1:0] multipland = 0;
	logic [DATA_WIDTH-1:0] add           ;
	logic [           2:0] p_d           ;
    logic [DATA_WIDTH-1:0] pp            ;
    logic                  d, s, a, c;
	logic [          2*DATA_WIDTH-1:0] p     = 0;
	logic [          2*DATA_WIDTH-1:0] shift    ;
	logic [          2*DATA_WIDTH-1:0] res      ;
	logic [$clog2(DATA_WIDTH/2 + 1):0] count = 0;

    // Do the Booth Encoder
    booth2encoder INST_BOOTH_ENCODE (p_d[0],p_d[1],p_d[2],d,s,a);

    // Generate the Partial Product
    booth2ppg #(DATA_WIDTH) INST_BOOTH_PPG (multipland,d,s,a,pp);

    // Shift the Results
    adder #(DATA_WIDTH) INST_ADDER (shift[2*DATA_WIDTH-1:DATA_WIDTH],pp+p_d[2],add);

    shifter #(2*DATA_WIDTH, 2) INST_SHIFTER (p,shift);

    assign res = {add, shift[DATA_WIDTH-1:0]};

    always_ff @(posedge clk) begin
        if (i_valid) begin
            p          <= {{DATA_WIDTH{1'b0}}, i_b};
            p_d        <= {i_b[1:0], 1'b0};
            multipland <= i_a;
            count      <= 0;
        end else begin
            p_d   <= p[3:1];
            p     <= res;
            count <= count + 1;
        end
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

module shifter #(DATA_WIDTH = 32, N = 2) (
	input  logic [DATA_WIDTH-1:0] a,
	output logic [DATA_WIDTH-1:0] b
);

	assign b = {{N{a[DATA_WIDTH-1]}}, a[DATA_WIDTH-1:N]};

endmodule

module booth2encoder (
    input  logic a1,
    input  logic a2,
    input  logic a3,
    output logic d ,
    output logic s ,
    output logic a
);

    assign d = a3;
    assign s = a3 ^ a2;
    assign a = a2 ^ a1;

endmodule // booth2encoder

module booth2ppg #(parameter DATA_WIDTH = 32) (
    input  logic [DATA_WIDTH-1:0] x,
    input  logic                  d,
    input  logic                  s,
    input  logic                  a,
    output logic [DATA_WIDTH-1:0] p
);

    genvar i;

    logic [DATA_WIDTH-1:0] neg_x;

    logic [DATA_WIDTH-1:0] d_res;
    logic [DATA_WIDTH-1:0] s_res;

    assign d_res[0] = d ? ~x[0] : x[0];
    assign s_res[0] = s ? d : '0;
    assign p[0]     = a ? d_res[0] : s_res[0];

    generate
        for (i = 1; i < DATA_WIDTH; i++) begin : GEN_MUX_PPG
            assign d_res[i] = d ? ~x[i] : x[i];
            assign s_res[i] = s ? d_res[i-1] : '0;
            assign p[i] = a ? d_res[i] : s_res[i];
        end
    endgenerate

endmodule // booth2ppg

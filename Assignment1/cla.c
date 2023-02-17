/*********************************************************************/
//
// Created by Chander Iyer and Neil McGlohon.
// 01/14/2022: Revised Version that fixes buffer overruns & mem leaks
//
/*********************************************************************/

#include "main.h"

//Touch these defines
#define input_size 1024 // hex digits 
#define block_size    8
#define verbose 1

//Do not touch these defines
#define digits (input_size+1)
#define bits (digits*4)
#define ngroups bits/block_size
#define nsections ngroups/block_size
#define nsupersections nsections/block_size

//Global definitions of the various arrays used in steps for easy access
int gi[bits] = {0};
int pi[bits] = {0};
int ci[bits] = {0};

int ggj[ngroups] = {0};
int gpj[ngroups] = {0};
int gcj[ngroups] = {0};

int sgk[nsections] = {0};
int spk[nsections] = {0};
int sck[nsections] = {0};

int ssgl[nsupersections] = {0} ;
int sspl[nsupersections] = {0} ;
int sscl[nsupersections] = {0} ;

int sumi[bits] = {0};

//Integer array of inputs in binary form
int* bin1=NULL;
int* bin2=NULL;

//Character array of inputs in hex form
char* hex1=NULL;
char* hex2=NULL;

void read_input()
{
	char* in1 = calloc(input_size+1, sizeof(char));
	char* in2 = calloc(input_size+1, sizeof(char));

	scanf("%s", in1);
	scanf("%s", in2);

	hex1 = grab_slice_char(in1,0,input_size+1);
	hex2 = grab_slice_char(in2,0,input_size+1);

	free(in1);
	free(in2);
}

void compute_gp()
{
	for(int i = 0; i < bits; i++)
	{
		gi[i] = bin1[i] & bin2[i];
		pi[i] = bin1[i] | bin2[i];
	}
}

//general function used to calculate the gp values for every level other than the lowest one. the 
//formula for a group of 4 bits is as follows: 
//ggj = gi+3 + pi+3gi+2 + pi+3pi+2gi+1 + pi+3pi+2pi+1gi
//gpj = pi+3pi+2pi+1pi 
//this can be expanded to calculate the values for any given block size
void general_compute_gp(int n, int* cg, int* bg, int* cp, int* bp) {
	for(int j = 0; j < n; j++)
	{
		//retrieving the appropriate block to generate higher level values for
		int jstart = j*block_size;
		int* g_group = grab_slice(bg,jstart,block_size);
		int* p_group = grab_slice(bp,jstart,block_size);

		//calculating g values
		int sum = 0;
		for(int i = 0; i < block_size; i++)
		{
			int mult = g_group[i]; //grabs the g_i term for the multiplication
			for(int ii = block_size-1; ii > i; ii--)
			{
				mult &= p_group[ii]; //grabs the p_i terms and multiplies it with the previously multiplied stuff (or the g_i term if first round)
			}
			sum |= mult; //sum up each of these things with an or
		}
		cg[j] = sum;

		//calculating p values
		int mult = p_group[0];
		for(int i = 1; i < block_size; i++)
		{
			mult &= p_group[i];
		}
		cp[j] = mult;

		// free from grab_slice allocation
		free(g_group);
		free(p_group);
	}
}

void compute_group_gp()
{
	//calls general compute gp function with appropriate arguments for given level
	general_compute_gp(ngroups, ggj, gi, gpj, pi);
}

void compute_section_gp()
{
	//calls general compute gp function with appropriate arguments for given level
	general_compute_gp(nsections, sgk, ggj, spk, gpj);
}

void compute_super_section_gp()
{
	//calls general compute gp function with appropriate arguments for given level
	general_compute_gp(nsupersections, ssgl, sgk, sspl, spk);
}

//follows the formula sscl = ssgl + sspl * sscl-1
void compute_super_section_carry()
{
	sscl[0] = ssgl[0] | (sspl[0] & 0); //first bits, no carry in at this point
	for (int l=1; l<nsupersections; l++) {
		int t = sspl[l] & sscl[l-1];
		sscl[l] = ssgl[l] | t;
	}
}

//general function to compute the carry bits for the current "level". takes the number of sections/groups, the current 
//level's g array, p array c array and the "above" level's c array
void general_compute_carry(int n, int* gs, int* ps, int* cs, int* acs) {
	for (int l=0; l<n; l++) {
		for (int k = l*block_size; k < (l+1)*block_size; k++) {
			int t = ps[k];

			if (k==0) t &= 0; //first bits, no carry in at this point
			else if (k % block_size == 0) t &= acs[l - 1]; //beginning of each section, we get the carry in from the "above" section 
			else t &= cs[k-1]; //for other carry bits that arent the above two cases, we look at the previous bit in the same section

			cs[k] = gs[k] | t;
		}
	}
}

void compute_section_carry()
{
	//calls general compute carry function with appropriate arguments for given level
	general_compute_carry(nsupersections, sgk, spk, sck, sscl); 
}

void compute_group_carry()
{
	//calls general compute carry function with appropriate arguments for given level
	general_compute_carry(nsections, ggj, gpj, gcj, sck);
}

void compute_carry()
{
	//calls general compute carry function with appropriate arguments for given level
	general_compute_carry(ngroups, gi, pi, ci, gcj);
}

//follows the formula sum[i] = a[i] xor b[i] xor c[i-1]
void compute_sum()
{
	sumi[0] = bin1[0] ^ bin2[0] ^ 0; //initial carry in bit is 0
	for (int i=1; i<bits; i++) {
		sumi[i] = bin1[i] ^ bin2[i] ^ ci[i-1]; 
	}
}

void cla()
{
	compute_gp();
	compute_group_gp();
	compute_section_gp();
	compute_super_section_gp();
	compute_super_section_carry();
	compute_section_carry();
	compute_group_carry();
	compute_carry();
	compute_sum();
}

int main(int argc, char *argv[])
{
  int randomGenerateFlag = 0;

  char* hexa=NULL;
  char* hexb=NULL;
  char* hexSum=NULL;
  char* int2str_result=NULL;  

	if (argc == 2) {
		if (strcmp(argv[1], "-r") == 0)
			randomGenerateFlag = 1;
	}

	if (randomGenerateFlag == 0)
	{
		read_input();
	}
	else
	{
		hex1 = generate_random_hex(input_size);
		hex2 = generate_random_hex(input_size);
	}

	hexa = prepend_non_sig_zero(hex1);
	hexb = prepend_non_sig_zero(hex2);
	hexa[digits] = '\0'; //double checking
	hexb[digits] = '\0';

	bin1 = gen_formated_binary_from_hex(hexa);
	bin2 = gen_formated_binary_from_hex(hexb);

	cla();

	int2str_result = int_to_string(sumi,bits);
	hexSum = revbinary_to_hex( int2str_result,bits);
	// hexSum = revbinary_to_hex(int_to_string(sumi,bits),bits);
	// free inputs fields allocated in read_input or gen random calls
	free(int2str_result);
	free(hex1);
	free(hex2);

	// free bin conversion of hex inputs
	free(bin1);
	free(bin2);

	// free memory from prepend call
	free(hexa);
	free(hexb);
	
	printf("%s\n",hexSum);

	free(hexSum);
	
	return 1;
}

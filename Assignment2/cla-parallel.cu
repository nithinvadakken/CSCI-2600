/*********************************************************************/
//
// 02/02/2023: Revised Version for 32M bit adder with 32 bit blocks
//
/*********************************************************************/

#include "main.h"

//Touch these defines
#define input_size 8388608 // hex digits 
#define block_size 32
#define verbose 0

//Do not touch these defines
#define digits (input_size+1)
#define bits (digits*4)
#define ngroups bits/block_size
#define nsections ngroups/block_size
#define nsupersections nsections/block_size
#define nsupersupersections nsupersections/block_size

//Global definitions of the various arrays used in steps for easy access
/***********************************************************************************************************/
// ADAPT AS CUDA managedMalloc memory - e.g., change to pointers and allocate in main function. 
/***********************************************************************************************************/
int* gi;
int* pi; 
int* ci;

int* ggj;
int* gpj;
int* gcj;

int* sgk;
int* spk;
int* sck;

int* ssgl;
int* sspl;
int* sscl;

int* ssspm;
int* sssgm;
int* ssscm;

int* dsumi;
int* dbin1;
int* dbin2;

//host side
int sumi[bits] = {0};
int sumrca[bits] = {0};

//Integer array of inputs in binary form
int* bin1=NULL;
int* bin2=NULL;

//Character array of inputs in hex form
char* hex1=NULL;
char* hex2=NULL;

void read_input()
{
	char* in1 = (char *)calloc(input_size+1, sizeof(char));
	char* in2 = (char *)calloc(input_size+1, sizeof(char));

	if( 1 != scanf("%s", in1))
	{
		printf("Failed to read input 1\n");
		exit(-1);
	}
	if( 1 != scanf("%s", in2))
	{
		printf("Failed to read input 2\n");
		exit(-1);
	}
	
	hex1 = grab_slice_char(in1,0,input_size+1);
	hex2 = grab_slice_char(in2,0,input_size+1);
	
	free(in1);
	free(in2);
}

/***********************************************************************************************************/
// ADAPT AS CUDA KERNEL 
/***********************************************************************************************************/

//This function computes the gs and ps for all bits
__global__ void compute_gp(const int* b1, const int* b2, int* g, int* p){
	size_t i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i>=bits) 
		return; //Throw out any aditional threads 
	g[i] = b1[i] & b2[i];
	p[i] = b1[i] | b2[i];
}

//general kernal used to calculate the gp values for every level other than the lowest one. the 
//formula for a group of 4 bits is as follows: 
//ggj = gi+3 + pi+3gi+2 + pi+3pi+2gi+1 + pi+3pi+2pi+1gi
//gpj = pi+3pi+2pi+1pi 
//this can be expanded to calculate the values for any given block size
__global__ void compute_general_gp(int nbit, const int* prevG, const int* prevP, int* curG, int* curP){
	size_t i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i>=nbit) 
		return; //Throw out any aditional threads 
	int start = i*block_size;

	//pointers to the groups that the thread is reducing
	const int* g_group = prevG+start;
	const int* p_group = prevP+start;
	
	//calculating g values
	int sum = 0;
	for(int j = 0; j < block_size; j++){
		int mult = g_group[j]; //grabs the g_i term for the multiplication
		for(int k = block_size-1; k > j; k--) {
			mult &= p_group[k]; //grabs the p_i terms and multiplies it with the previously multiplied stuff (or the g_i term if first round)
		}
		sum |= mult; //sum up each of these things with an or
	}
	curG[i] = sum;

	//calculating p values
	int mult = p_group[0];
	for(int j = 1; j < block_size; j++) {
		mult &= p_group[j];
	}
	curP[i] = mult;
}

//calls general compute gp function with appropriate arguments for given level
void compute_group_gp(){
	compute_general_gp<<<ngroups,32>>>(ngroups, gi, pi, ggj, gpj);
}

//calls general compute gp function with appropriate arguments for given level
void compute_section_gp() {
	compute_general_gp<<<nsections,32>>>(nsections, ggj, gpj, sgk, spk);
}

//calls general compute gp function with appropriate arguments for given level
void compute_super_section_gp() {
	compute_general_gp<<<nsupersections,32>>>(nsupersections, sgk, spk, ssgl, sspl);
}

void compute_super_super_section_gp() {
	compute_general_gp<<<nsupersupersections,32>>>(nsupersupersections, ssgl, sspl, sssgm, ssspm);
}

//general function to compute the carry bits for the current "level". takes the number of sections/groups, the current 
//level's g array, p array c array and the "above" level's c array
__global__ void compute_general_carry(int n, const int* prevG, const int* prevP, const int* curC, int* prevC){
	size_t i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i>=n) return;
	int carry = i == 0 ? 0 : curC[i-1];
	for(int j = i*block_size; j < (i+1)*block_size; j++){
		prevC[j] = prevG[j] | (prevP[j] & carry);
		carry = prevC[j];
	}
}

//calls general compute carry function with appropriate arguments for given level
void compute_super_super_section_carry()
{
	compute_general_carry<<<1,32>>>(1, sssgm, ssspm, nullptr, ssscm);
}

//calls general compute carry function with appropriate arguments for given level
void compute_super_section_carry()
{
	compute_general_carry<<<nsupersupersections,32>>>(nsupersupersections, ssgl, sspl, ssscm, sscl);
}

//calls general compute carry function with appropriate arguments for given level
void compute_section_carry()
{
	compute_general_carry<<<nsupersections,32>>>(nsupersections, sgk, spk, sscl, sck);
}

//calls general compute carry function with appropriate arguments for given level
void compute_group_carry()
{
	compute_general_carry<<<nsections,32>>>(nsections, ggj, gpj, sck, gcj);
}

//calls general compute carry function with appropriate arguments for given level
void compute_carry()
{
	compute_general_carry<<<ngroups,32>>>(ngroups, gi, pi, gcj, ci);
}

__global__ void compute_sum(int* sum, const int* b1, const int* b2, const int* c){
	size_t i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i>=bits) return;
	int carry = i == 0 ? 0 : c[i-1];
	if (i==0) {carry = 0;}
	else {carry = c[i-1];}
	sum[i] = b1[i] ^ b2[i] ^ carry;
}

void cla()
{
	//allocating all the memory on the GPU
	cudaMallocManaged((void**)&gi, bits*sizeof(int));
	cudaMallocManaged((void**)&pi, bits*sizeof(int));
	cudaMallocManaged((void**)&ci, bits*sizeof(int));
	cudaMallocManaged((void**)&ggj, ngroups*sizeof(int));
	cudaMallocManaged((void**)&gpj, ngroups*sizeof(int));
	cudaMallocManaged((void**)&gcj, ngroups*sizeof(int));
	cudaMallocManaged((void**)&sgk, nsections*sizeof(int));
	cudaMallocManaged((void**)&spk, nsections*sizeof(int));
	cudaMallocManaged((void**)&sck, nsections*sizeof(int));
	cudaMallocManaged((void**)&ssgl, nsupersections*sizeof(int));
	cudaMallocManaged((void**)&sspl, nsupersections*sizeof(int));
	cudaMallocManaged((void**)&sscl, nsupersections*sizeof(int));
	cudaMallocManaged((void**)&sssgm, nsupersupersections*sizeof(int));
	cudaMallocManaged((void**)&ssspm, nsupersupersections*sizeof(int));
	cudaMallocManaged((void**)&ssscm, nsupersupersections*sizeof(int));

	//getting the gp values for the lowest level
	compute_gp<<<bits,32>>>(dbin1, dbin2, gi, pi);

	//each of these functions will call the general versions which will take care of everything
	//computing gps for each level
	compute_group_gp();
	compute_section_gp();
	compute_super_section_gp();
	compute_super_super_section_gp();

	//computing carries for each level
	compute_super_super_section_carry();
	compute_super_section_carry();
	compute_section_carry();
	compute_group_carry();
	compute_carry();

	//computing the sum
	compute_sum<<<bits,32>>>(dbin1, dbin2, ci, dsumi);

	cudaDeviceSynchronize();
}

void ripple_carry_adder()
{
	int clast=0, cnext=0;

	for(int i = 0; i < bits; i++)
		{
			cnext = (bin1[i] & bin2[i]) | ((bin1[i] | bin2[i]) & clast);
			sumrca[i] = bin1[i] ^ bin2[i] ^ clast;
			clast = cnext;
		}
}

void check_cla_rca()
{
	for(int i = 0; i < bits; i++)
		{
			if( sumrca[i] != sumi[i] )
	{
		printf("Check: Found sumrca[%d] = %d, not equal to sumi[%d] = %d - stopping check here!\n",
		 i, sumrca[i], i, sumi[i]);
		printf("bin1[%d] = %d, bin2[%d]=%d, gi[%d]=%d, pi[%d]=%d, ci[%d]=%d, ci[%d]=%d\n",
		 i, bin1[i], i, bin2[i], i, gi[i], i, pi[i], i, ci[i], i-1, ci[i-1]);
		return;
	}
		}
	printf("Check Complete: CLA and RCA are equal\n");
}

int main(int argc, char *argv[])
{
	int randomGenerateFlag = 1;
	int deterministic_seed = (1<<30) - 1;
	char* hexa=NULL;
	char* hexb=NULL;
	char* hexSum=NULL;
	char* int2str_result=NULL;
	unsigned long long start_time=clock_now(); // dummy clock reads to init
	unsigned long long end_time=clock_now();   // dummy clock reads to init

	if( nsupersupersections != block_size )
		{
			printf("Misconfigured CLA - nsupersupersections (%d) not equal to block_size (%d) \n",
			 nsupersupersections, block_size );
			return(-1);
		}
	
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
			srand( deterministic_seed );
			hex1 = generate_random_hex(input_size);
			hex2 = generate_random_hex(input_size);
		}
	
	hexa = prepend_non_sig_zero(hex1);
	hexb = prepend_non_sig_zero(hex2);
	hexa[digits] = '\0'; //double checking
	hexb[digits] = '\0';
	
	bin1 = gen_formated_binary_from_hex(hexa);
	bin2 = gen_formated_binary_from_hex(hexb);

	start_time = clock_now();
	cla();
	end_time = clock_now();

	printf("CLA Completed in %llu cycles\n", (end_time - start_time));

	start_time = clock_now();
	ripple_carry_adder();
	end_time = clock_now();

	printf("RCA Completed in %llu cycles\n", (end_time - start_time));

	check_cla_rca();

	if( verbose==1 )
		{
			int2str_result = int_to_string(sumi,bits);
			hexSum = revbinary_to_hex( int2str_result,bits);
		}

	// free inputs fields allocated in read_input or gen random calls
	free(int2str_result);
	free(hex1);
	free(hex2);
	
	// free bin conversion of hex inputs
	free(bin1);
	free(bin2);
	
	if( verbose==1 )
		{
			printf("Hex Input\n");
			printf("a   ");
			print_chararrayln(hexa);
			printf("b   ");
			print_chararrayln(hexb);
		}
	
	if ( verbose==1 )
		{
			printf("Hex Return\n");
			printf("sum =  ");
		}
	
	// free memory from prepend call
	free(hexa);
	free(hexb);

	if( verbose==1 )
		printf("%s\n",hexSum);
	
	free(hexSum);
	
	return 0;
}

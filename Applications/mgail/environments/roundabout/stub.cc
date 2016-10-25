#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>

#define SCAN_BATCH 1
#define Nt 5
#define INPUT_SIZE SCAN_BATCH * (Nt * 3 + 3)
#define OUTPUT_SIZE SCAN_BATCH * (Nt * 2 + 2)

int main(){

const char * input_fifo = "/tmp/input_fifo";
const char * output_fifo = "/tmp/output_fifo";

int fd_i = open(input_fifo, O_WRONLY);
int fd_o = open(output_fifo, O_RDONLY);

float input_buffer[INPUT_SIZE];
float output_buffer[OUTPUT_SIZE];

float state_ [12] = {2.777781, -14.074073,   5.285965,   5.302605,   8.4245  ,
         5.5198  ,   6.882078,  20.201031,  30.776821, -21.141763,
        -4.579117,   6.594841};

float is_aggressive [5] = {0,  0,  0,  0,  0};

float actions[38] = {0.000000e+00,   0.000000e+00,   0.000000e+00,   0.000000e+00,
         0.000000e+00,   0.000000e+00,   0.000000e+00,   0.000000e+00,
         0.000000e+00,   0.000000e+00,   0.000000e+00,   0.000000e+00,
         0.000000e+00,   0.000000e+00,   0.000000e+00,   0.000000e+00,
         0.000000e+00,   0.000000e+00,   0.000000e+00,   0.000000e+00,
        -3.862381e-05,   0.000000e+00,   0.000000e+00,   0.000000e+00,
        -2.499999e+01,   0.000000e+00,   0.000000e+00,   0.000000e+00,
         0.000000e+00,   0.000000e+00,   0.000000e+00,   0.000000e+00,
         3.263445e+01,   8.811829e+00,   1.033565e+01,   1.227476e+01,
         1.471409e+01,   1.775795e+01};

 /*send input*/
 for (int j=0; j<38 ; j++){

  printf("j: %d: ", j);

     for (int i=0; i < 12 ; i++){
        input_buffer[i] = state_[i];
        printf("%f, ", input_buffer[i]);
    }

     for (int i=0; i < 5 ; i ++){
        input_buffer[i+12] = is_aggressive[i];
        printf("%f, ", input_buffer[i+12]);
    }

    input_buffer[17] = actions[j];
    printf("%f", input_buffer[17]);

  int n_w = write(fd_i, input_buffer, sizeof(input_buffer));

  /*get output*/
  int n_r = read(fd_o, output_buffer, sizeof(output_buffer));

     for (int k=0; k < OUTPUT_SIZE ; k ++){
        state_[k] = output_buffer[k];
//        printf("%f, ", state_[k]);
    }

    printf("\n");
  }
}
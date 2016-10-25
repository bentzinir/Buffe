#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>

#define SCAN_BATCH 1
#define INPUT_SIZE SCAN_BATCH * 6
#define OUTPUT_SIZE SCAN_BATCH * 4

#define DT 0.15
#define V_MAX 0.5


int main()
{
    printf("Ready...\n");

    int fd_i;
    int fd_o;

    remove( "/tmp/input_fifo" );
    remove( "/tmp/output_fifo" );

    const char * input_fifo = "/tmp/input_fifo";
    const char * output_fifo = "/tmp/output_fifo";


    /* create the FIFO (named pipe) */
    mkfifo(input_fifo, 0666);
    mkfifo(output_fifo, 0666);

    fd_i = open(input_fifo, O_RDONLY);
    fd_o = open(output_fifo, O_WRONLY);

    // Declarations
    // inputs
    float input_buffer[INPUT_SIZE];
    float v_x_;
    float v_y_;
    float x_;
    float y_;
    float a_x;
    float a_y;

    // outputs
    float output_buffer[OUTPUT_SIZE];
    float v_x;
    float v_y;
    float x;
    float y;

    while (1) {

        float input_buffer[INPUT_SIZE];

        // read all states at once
        int n_r = read(fd_i, input_buffer, sizeof(input_buffer));

        // initialize pointers for input_buffer / output_buffer
        int i_ptr = 0;
        int o_ptr = 0;

        /* state:
         v_x_: 1
         v_y_: 1
         x_: 1
         y_: 1
         a_x: 1
         a_y: 1
        */

        for (int b=0; b< SCAN_BATCH ; b++){
            // 1. read state_b
            v_x_ = input_buffer[i_ptr++];
            v_y_ = input_buffer[i_ptr++];
            x_ = input_buffer[i_ptr++];
            y_ = input_buffer[i_ptr++];
            a_x = input_buffer[i_ptr++];
            a_y = input_buffer[i_ptr++];

            // 2. step
            v_x = v_x_ + DT * a_x;
            v_y = v_y_ + DT * a_y;

            v_x = (v_x > V_MAX) ? V_MAX : (v_x < -V_MAX) ? -V_MAX : v_x;
            v_y = (v_y > V_MAX) ? V_MAX : (v_y < -V_MAX) ? -V_MAX : v_y;

            x = x_ + DT * v_x;
            y = y_ + DT * v_y;

            // 3. write state_b to output_buffer
            output_buffer[o_ptr++] = v_x;
            output_buffer[o_ptr++] = v_y;
            output_buffer[o_ptr++] = x;
            output_buffer[o_ptr++] = y;

        }// for (int b=0; b< BATCH_SIZE ...

        // 4. write all states to output_buffer at once
        int n_w = write(fd_o, output_buffer, sizeof(output_buffer));
    }

    close(fd_i);
    close(fd_o);

    /* remove the FIFO */
    unlink(input_fifo);
    unlink(output_fifo);

    return 0;
}
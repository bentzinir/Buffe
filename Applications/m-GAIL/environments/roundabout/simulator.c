#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>

#define SCAN_BATCH 1
#define Nt 5
#define INPUT_SIZE SCAN_BATCH * (Nt * 4 + 3)
#define OUTPUT_SIZE SCAN_BATCH * (Nt * 3 + 2)

#define FPS 9
#define DT 1./FPS
#define PI 3.14159265359
#define R 10

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
    float v_h_;
    float x_h_;
    float v_t_[Nt];
    float x_t_[Nt];
    float a_t_[Nt];
    float is_aggressive[Nt];
    float action;
    // outputs
    float output_buffer[OUTPUT_SIZE];
    float v_h;
    float x_h;
    float v_t[Nt];
    float x_t[Nt];
    float a_t[Nt];

    while (1) {

        // read all states at once
        int n_r = read(fd_i, input_buffer, sizeof(input_buffer));

        // initialize pointers for input_buffer / output_buffer
        int i_ptr = 0;
        int o_ptr = 0;

        /* state:
        v_h_: 1
        x_h_: 1
        v_t_: Nt
        x_t_:Nt
        a_t_:Nt
        is_aggressive: Nt
        a: 1
        */

        for (int b=0; b< SCAN_BATCH ; b++){
            // 1. read state_b
            v_h_ = input_buffer[i_ptr++];
            x_h_ = input_buffer[i_ptr++];

            for (int i=0; i<Nt; i++){
                v_t_[i] = input_buffer[i_ptr++];
            }

            for (int i=0; i<Nt; i++){
                x_t_[i] = input_buffer[i_ptr++];
            }

            for (int i=0; i<Nt; i++){
                a_t_[i] = input_buffer[i_ptr++];
            }

            for (int i=0; i<Nt; i++){
                is_aggressive[i] = input_buffer[i_ptr++];
            }

             action = input_buffer[i_ptr++];

            // 2. increment host
            v_h = v_h_ + DT * action;

            // clip host speed to the section [0,20]
             v_h = (v_h < 0) ? 0 : (v_h > 20) ? 20 : v_h;
//            v_h = (v_h<0) ? 0 : v_h;

            x_h = x_h_ + DT * v_h;

            //
            //x_h = (x_h >= R * PI) ? x_h - 2 * PI * R : x_h;
            x_h = (x_h >= 0.5 * R * PI) ? 0.5 * R * PI : x_h;

            // 3. increment targets
            for (int t=0 ; t<Nt ; t++){

                float next_x_t_ = x_t_[(t+1)%Nt];

                float relx = next_x_t_ - x_t_[t];

                // fix the jump between -pi and +pi
                relx = (relx>=0) ? relx : 2*PI*R + relx;

                float relx_to_host = x_h_ - x_t_[t];

                bool is_host_approaching = (x_h_ > -7) &
                                           (x_h_ < 2) &
                                           (x_t_[t] > -2*PI*R/4) &
                                           (x_t_[t] < 0) &
                                           ( (next_x_t_ > 0) |
                                             (next_x_t_ < x_t_[t]) &
                                             (relx > 2) );

                bool is_host_on_route = (x_h_ > 0);

                bool is_host_cipv = (x_h_ > x_t_[t]) &
                                    (relx_to_host < relx);

                float accel_default = 5*(relx - 2*v_t_[t]);

                float accel_host_on_route = is_host_cipv ? (5 * (relx_to_host - 1.5 * v_t_[t]) ) : accel_default;

                float accel_is_aggressive = 3*(relx_to_host - 1.5*v_t_[t]);

                accel_is_aggressive = (accel_is_aggressive > 3) ? 3 : accel_is_aggressive;

                float accel_not_aggressive = relx_to_host - 1.5 * v_t_[t];

                float accel_host_approaching = is_aggressive ? accel_is_aggressive : accel_not_aggressive;

                float accel_host_not_approaching = is_host_on_route ? accel_host_on_route : accel_default;

                float accel = is_host_approaching ? accel_host_approaching : accel_host_not_approaching;

                v_t[t] = v_t_[t] + DT * accel;

                v_t[t] = v_t[t] < 0 ? 0 : v_t[t];

                x_t[t] = x_t_[t] + DT * v_t[t];

                x_t[t] = (x_t[t] > R * PI) ? x_t[t] - 2 * PI * R : x_t[t];

                a_t[t] = accel;
            }

            // 4. write state_b to output buffer
            // 4.1 host
            output_buffer[o_ptr++] = v_h;
            output_buffer[o_ptr++] = x_h;

            // 4.2 target
            for (int i=0; i<Nt; i++){
                output_buffer[o_ptr++] = v_t[i];
            }

            for (int i=0; i<Nt; i++){
                output_buffer[o_ptr++] = x_t[i];
            }

             for (int i=0; i<Nt; i++){
                output_buffer[o_ptr++] = a_t[i];
            }

        } // for (int b=0; b< BATCH_SIZE ...

        // 5. write all states to output_buffer at once
        int n_w = write(fd_o, output_buffer, sizeof(output_buffer));
    }

    close(fd_i);
    close(fd_o);

    /* remove the FIFO */
    unlink(input_fifo);
    unlink(output_fifo);

    return 0;
}
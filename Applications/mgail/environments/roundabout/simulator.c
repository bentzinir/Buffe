#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>

// state_: np.concatenate([[x_h, x_ct, x_h_, x_ct_, x_h__, x_ct__], v_t, x_t, a_t, self.is_aggressive, [ct_ind]])
#define SCAN_BATCH 1
#define NT 5
#define ACTION_SIZE 1
#define STATE_SIZE 27
#define INPUT_SIZE SCAN_BATCH * ( STATE_SIZE + ACTION_SIZE)
#define OUTPUT_SIZE SCAN_BATCH * (STATE_SIZE)

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
    float x_H_[6];
    float x_h_;
    float v_t_[NT];
    float x_t_[NT];
    float a_t_[NT];
    float is_aggressive[NT];
    int ct_ind; //index of closest vehicle
    float action;
    // outputs
    float output_buffer[OUTPUT_SIZE];
    float x_H[6];
    float x_h;
    float v_t[NT];
    float x_t[NT];
    float a_t[NT];

    while (1) {

        // read all states at once
        int n_r = read(fd_i, input_buffer, sizeof(input_buffer));

        // initialize pointers for input_buffer / output_buffer
        int i_ptr = 0;
        int o_ptr = 0;

        /* state:
        x_H_: 6: [x_h, x_ct, x_h_, x_ct_, x_h__, x_ct__]
        v_t_: NT
        x_t_:NT
        a_t_:NT
        is_aggressive: NT
        ct_ind: 1
        action: 1: dx
        */

        for (int b=0; b< SCAN_BATCH ; b++){
            // 1. read state_b
            for (int i=0; i<6; i++){
                x_H_[i] = input_buffer[i_ptr++];
            }

            for (int i=0; i<NT; i++){
                v_t_[i] = input_buffer[i_ptr++];
            }

            for (int i=0; i<NT; i++){
                x_t_[i] = input_buffer[i_ptr++];
            }

            for (int i=0; i<NT; i++){
                a_t_[i] = input_buffer[i_ptr++];
            }

            for (int i=0; i<NT; i++){
                is_aggressive[i] = input_buffer[i_ptr++];
            }

             ct_ind = (int)input_buffer[i_ptr++];
             action = input_buffer[i_ptr++];

            // 2. increment host
            action = (action>2.5) ? 2.5 : (action<0.) ? 0. : action;
            x_h_ = x_H_[0];

            //v_h = v_h_ + DT * action;

            // clip host speed to the section [0,20]
            // v_h = (v_h < 0) ? 0 : (v_h > 20) ? 20 : v_h;

            //x_h = x_h_ + DT * v_h;
            x_h = x_h_ + action;

            x_h = (x_h >= 0.5 * R * PI) ? 0.5 * R * PI : (x_h < - R * PI) ? - R * PI : x_h;
//            x_h = (x_h >= 0) ? 0 : (x_h <= - R * PI) ? - R * PI : x_h;

            // 3. increment targets
            for (int t=0 ; t<NT ; t++){

                float next_x_t_ = x_t_[(t+1)%NT];

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

            // 4. update history:
            // 4.1 x_H[0:1]
            x_H[0] = x_h;
            x_H[1] = x_t[ct_ind];

            // 4.2 x_H[2:6] = x_H_[0:4]
            for (int i=0; i<4; i++){
                x_H[i+2] = x_H_[i];
            }

            // 5. write state_b to output buffer
            // 5.2 x_H
             for (int i=0; i<6; i++){
                output_buffer[o_ptr++] = x_H[i];
            }

            // 4.3 targets
            // 4.3.1 speeds
            for (int i=0; i<NT; i++){
                output_buffer[o_ptr++] = v_t[i];
            }

            // 4.3.2 positions
            for (int i=0; i<NT; i++){
                output_buffer[o_ptr++] = x_t[i];
            }

            // 4.3.3 accelerations
             for (int i=0; i<NT; i++){
                output_buffer[o_ptr++] = a_t[i];
            }

            // 4.3.4 is_aggressive
             for (int i=0; i<NT; i++){
                output_buffer[o_ptr++] = is_aggressive[i];
            }

            // 4.3.5 ct_ind
            output_buffer[o_ptr++] = (float)ct_ind;

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
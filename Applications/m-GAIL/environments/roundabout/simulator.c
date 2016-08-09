#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>

#define Nt 5
#define INPUT_SIZE Nt * 4 + 3
#define OUTPUT_SIZE Nt * 3 + 2

#define FPS 15
#define DT 1./FPS
#define V0 5.
#define PI 3.14159265359
#define HOST_LENGTH 144.0
#define R 10

int main()
{
    printf("Ready...\n");

    int fd_i;
    int fd_o;

    const char * input_fifo = "/tmp/input_fifo";
    const char * output_fifo = "/tmp/output_fifo";

    /* create the FIFO (named pipe) */
    mkfifo(input_fifo, 0666);
    mkfifo(output_fifo, 0666);

    fd_i = open(input_fifo, O_RDONLY);
    fd_o = open(output_fifo, O_WRONLY);

    while (1) {

        float input_buffer[INPUT_SIZE];

        int n_r = read(fd_i, input_buffer, sizeof(input_buffer));

        /* state:
        v_h_: 1
        x_h_: 1
        v_t_: Nt
        x_t_:Nt
        a_t_:Nt
        is_aggressive: Nt
        a: 1
        */

        int fld_ptr = 0;

        float v_h_ = input_buffer[fld_ptr++];
        float x_h_ = input_buffer[fld_ptr++];

        float v_t_[Nt];
        for (int i=0; i<Nt; i++){
            v_t_[i] = input_buffer[fld_ptr++];
        }

        float x_t_[Nt];
        for (int i=0; i<Nt; i++){
            x_t_[i] = input_buffer[fld_ptr++];
        }

        float a_t_[Nt];
        for (int i=0; i<Nt; i++){
            a_t_[i] = input_buffer[fld_ptr++];
        }

        float is_aggressive[Nt];
        for (int i=0; i<Nt; i++){
            is_aggressive[i] = input_buffer[fld_ptr++];
        }

        float action = input_buffer[fld_ptr++];

        float v_t[Nt];
        float x_t[Nt];
        float a_t[Nt];

        // host
         float v_h = v_h_ + DT * action;

         // clip host speed to the section [0,v0]
         v_h = (v_h > 3 * V0) ? v_h : (v_h < 0) ? 0 : v_h;

         float x_h = x_h_ + DT * v_h;

         x_h = (x_h >= PI) ? x_h - 2 * PI * R : x_h;

        // targets
        for (int i=0 ; i<Nt ; i++){

            float next_x_t_ = x_t_[(i+1)%Nt];

            float relx = next_x_t_ - x_t_[i];

            // fix the jump between -pi and +pi
            relx = (relx>=0) ? relx : 2*PI*R + relx;

            float relx_to_host = x_h_ - x_t_[i];

            bool is_host_cipv = (x_h_ > -0.5 * HOST_LENGTH) &
                                (x_h_ > x_t_[i]) &
                                (relx_to_host < relx);

            // If host CIPV - Change relx to him
            if (is_host_cipv) {
                relx = (x_h_>0) ? x_h_ - x_t_[i] : -x_t_[i];
            }

            bool is_host_approaching = (x_h_ > -1.5 * HOST_LENGTH) &
                                       (x_h_ <= -0.5 * HOST_LENGTH) &
                                       (x_t_[i] < 0) &
                                       (x_t_[i] > -0.25 * 2*PI*R) &
                                       ( (next_x_t_ > 0) | (next_x_t_ < x_t_[i]) );

            float accel_default = 5*(relx - 2*v_t_[i]);

            // accel_is_aggressive = (3 - v_t_)/self.dt
            float accel_is_aggressive = 3*(relx_to_host - 1.5*v_t_[i]);

            accel_is_aggressive = (accel_is_aggressive > 3) ? 3 : accel_is_aggressive;

            float accel_not_aggressive = (0.5 - v_t_[i])/DT;

            float accel_host_approaching = is_aggressive ? accel_is_aggressive : accel_not_aggressive;

            float accel = is_host_approaching ? accel_host_approaching : accel_default;

            v_t[i] = v_t_[i] + DT * accel;

            v_t[i] = v_t[i] < 0 ? 0 : v_t[i];

            x_t[i] = x_t_[i] + DT * v_t[i];

            a_t[i] = v_t[i] - v_t_[i];
        }


        /* write new state back to tf*/
        float output_buffer[OUTPUT_SIZE];

        fld_ptr = 0;

        // host
        output_buffer[fld_ptr++] = v_h;
        output_buffer[fld_ptr++] = x_h;

        // target
        for (int i=0; i<Nt; i++){
            output_buffer[fld_ptr++] = v_t[i];
        }

        for (int i=0; i<Nt; i++){
            output_buffer[fld_ptr++] = x_t[i];
        }

         for (int i=0; i<Nt; i++){
            output_buffer[fld_ptr++] = a_t[i];
        }

        int n_w = write(fd_o, output_buffer, sizeof(output_buffer));
    }

    close(fd_i);
    close(fd_o);

    /* remove the FIFO */
    unlink(input_fifo);
    unlink(output_fifo);

    return 0;
}
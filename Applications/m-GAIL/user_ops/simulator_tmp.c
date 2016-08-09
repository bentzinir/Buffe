#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#define INPUT_SIZE 3
#define OUTPUT_SIZE 2

#define DT 0.15
#define V_MAX 0.5


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
        printf("sizeof input_buffer: %d, read %d\n", (int)sizeof(input_buffer), n_r);

        float v_ = input_buffer[0];
        float x_ = input_buffer[1];
        float a = input_buffer[2];

        printf("v_: %f, x_: %f, a: %f\n", v_, x_, a);

        /* simulator step */
        float v_x = v_ + DT * a;
        float v_y = v_ + DT * a;

        //v_x = (v_x > V_MAX) ? V_MAX : (v_x < -V_MAX) ? -V_MAX : v_x;
        //v_y = (v_y > V_MAX) ? V_MAX : (v_y < -V_MAX) ? -V_MAX : v_y;

        float x = x_ + DT * v_x;
        float y = x_ + DT * v_y;

        /* write new state back to tf*/
        float output_buffer[OUTPUT_SIZE];

        output_buffer[0] = v_x;
        output_buffer[1] = v_y;

        printf("after: v_x: %f, v_y: %f\n", v_x, v_y);

        int n_w = write(fd_o, output_buffer, sizeof(output_buffer));
        printf("wrote: %d\n", n_w);
    }

    close(fd_i);
    close(fd_o);

    /* remove the FIFO */
    unlink(input_fifo);
    unlink(output_fifo);

    return 0;
}
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#define INPUT_SIZE 6
#define OUTPUT_SIZE 4

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

        //char input_buffer_char[sizeof(input_buffer)];
        //int n_r = read(fd_i, input_buffer_char, sizeof(input_buffer));
        int n_r = read(fd_i, input_buffer, sizeof(input_buffer));
        //printf("sizeof input_buffer: %d, read %d\n", (int)sizeof(input_buffer), n_r);

        //memcpy(input_buffer, input_buffer_char, sizeof(input_buffer));

        float v_x_ = input_buffer[0];
        float v_y_ = input_buffer[1];
        float x_ = input_buffer[2];
        float y_ = input_buffer[3];
        float a_x = input_buffer[4];
        float a_y = input_buffer[5];

        //printf("v_x: %f, v_y: %f, x_: %f, y_: %f, a_x: %f, a_y: %f\n", v_x_, v_y_, x_, y_, a_x, a_y);
        /* simulator step */
        float v_x = v_x_ + DT * a_x;
        float v_y = v_y_ + DT * a_y;

        v_x = (v_x > V_MAX) ? V_MAX : (v_x < -V_MAX) ? -V_MAX : v_x;
        v_y = (v_y > V_MAX) ? V_MAX : (v_y < -V_MAX) ? -V_MAX : v_y;

        float x = x_ + DT * v_x;
        float y = y_ + DT * v_y;

        /* write new state back to tf*/
        float output_buffer[OUTPUT_SIZE];

        output_buffer[0] = v_x;
        output_buffer[1] = v_y;
        output_buffer[2] = x;
        output_buffer[3] = y;

        //printf("after: v_x: %f, v_y: %f, x_: %f, y_: %f\n", v_x, v_y, x, y);

        //char output_buffer_char[sizeof(output_buffer)];

        //memcpy(input_buffer_char, input_buffer, sizeof(input_buffer));

        //int n_w = write(fd_o, output_buffer_char, sizeof(output_buffer));
        int n_w = write(fd_o, output_buffer, sizeof(output_buffer));
        //printf("wrote: %d\n", n_w);
    }

    close(fd_i);
    close(fd_o);

    /* remove the FIFO */
    unlink(input_fifo);
    unlink(output_fifo);

    return 0;
}
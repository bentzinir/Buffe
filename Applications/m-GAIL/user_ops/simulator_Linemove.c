#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <stdio.h>

#define INPUT_SIZE 3
#define OUTPUT_SIZE 2

#define DT 0.15
#define V_MAX 0.5

int main()
{
    int fd_i;
    int fd_o;

    const char * input_fifo = "/tmp/input_fifo";
    const char * output_fifo = "/tmp/output_fifo";

    float input_buffer[INPUT_SIZE];
    float output_buffer[OUTPUT_SIZE];

    /* create the FIFO (named pipe) */
    mkfifo(input_fifo, 0666);
    mkfifo(output_fifo, 0666);

    fd_i = open(input_fifo, O_RDONLY);
    fd_o = open(output_fifo, O_WRONLY);

    while (1) {

        float input[INPUT_SIZE];
        read(fd_i, input_buffer, (INPUT_SIZE) * sizeof(float));
        float v_ = input_buffer[0];
        float x_ = input_buffer[1];
        float a = input_buffer[2];

        /* simulator step */
        float v = v_ + DT * a;

        v = (v > V_MAX) ? V_MAX : (v < -V_MAX) ? -V_MAX : v;

        float x = x_ + DT * v;

        /* write new state back to tf*/
        output_buffer[0] = v;
        output_buffer[1] = x;
        write(fd_o, output_buffer, sizeof(output_buffer));
    }

    close(fd_i);
    close(fd_o);

    /* remove the FIFO */
    unlink(input_fifo);
    unlink(output_fifo);

    return 0;
}
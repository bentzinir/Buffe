#include "tensorflow/core/framework/op.h"

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

REGISTER_OP("SimStep")
    .Input("state_: float32")
    .Output("state: float32");

#define INPUT_SIZE 3
#define OUTPUT_SIZE 2

#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

class SimStepOp : public OpKernel {
 public:

    const char * input_fifo = "/tmp/input_fifo";
    const char * output_fifo = "/tmp/output_fifo";

    int fd_i = open(input_fifo, O_WRONLY);
    int fd_o = open(output_fifo, O_RDONLY);

    float input_buffer[INPUT_SIZE];
    float output_buffer[OUTPUT_SIZE];

  explicit SimStepOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    //auto input = input_tensor.flat<int32>();
    auto input = input_tensor.flat<float>();

//        float v_x_ = input(0);
//        float v_y_ = input(1);
//        float x_ = input(2);
//        float y_ = input(3);
//        float a_x = input(4);
//        float a_y = input(5);
//
//        printf("v_x: %f, v_y: %f, x_: %f, y_: %f, a_x: %f, a_y: %f\n", v_x_, v_y_, x_, y_, a_x, a_y);

    for (int i=0; i < INPUT_SIZE ; i ++){
        input_buffer[i] = input(i);
    }
        float v_x_ = input_buffer[0];
        float v_y_ = input_buffer[1];
        float x_ = input_buffer[2];
//        float y_ = input_buffer[3];
//        float a_x = input_buffer[4];
//        float a_y = input_buffer[5];

        printf("send: v_x: %f, v_y: %f, x_: %f\n", v_x_, v_y_, x_);

     /*send input*/
      int n_w = write(fd_i, input_buffer, sizeof(input_buffer));
      //int n_w = write(fd_i, input_buffer, sizeof(input_buffer));
      printf("wrote %d\n", n_w);

     /*get output*/
     int n_r = read(fd_o, output_buffer, sizeof(output_buffer));
     printf("read: %d\n", n_r);
     printf("read: v_x: %f, v_y: %f, x_: %f\n", output_buffer[0], output_buffer[1], output_buffer[2]);

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    auto output = output_tensor->flat<float>();

    for (int j=0; j < OUTPUT_SIZE ; j ++){
        output(j) = output_buffer[j];
    }
  }
  
};

REGISTER_KERNEL_BUILDER(Name("SimStep").Device(DEVICE_CPU), SimStepOp);
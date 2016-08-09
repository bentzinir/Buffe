#include "tensorflow/core/framework/op.h"

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

REGISTER_OP("SimStep")
    .Input("state_: float32")
    .Output("state: float32");

#define INPUT_SIZE 6
#define OUTPUT_SIZE 4

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

    for (int i=0; i < INPUT_SIZE ; i ++){
        input_buffer[i] = input(i);
    }

     /*send input*/
      int n_w = write(fd_i, input_buffer, sizeof(input_buffer));

     /*get output*/
     int n_r = read(fd_o, output_buffer, sizeof(output_buffer));

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
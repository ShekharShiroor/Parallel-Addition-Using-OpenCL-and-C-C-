#include <wb.h>
#include <CL/opencl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

//@@ Write the OpenCL kernel
const char *kernelSource = "";

int main(int argc, char *argv[]) {
  wbArg_t args;
  int inputLength;
  int inputLengthBytes;
  float *hostInput1;
  float *hostInput2;
  float *hostOutput;
  cl_mem deviceInput1;
  cl_mem deviceInput2;
  cl_mem deviceOutput;
  deviceInput1 = NULL;
  deviceInput2 = NULL;
  deviceOutput = NULL;


  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput1 =
      (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostInput2 =
      (float *)wbImport(wbArg_getInputFile(args, 1), &inputLength);
  inputLengthBytes = inputLength * sizeof(float);
  hostOutput       = (float *)malloc(inputLengthBytes);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The input length is ", inputLength);
  wbLog(TRACE, "The input size is ", inputLengthBytes, " bytes");


  cl_int clerr = CL_SUCCESS;
  cl_platform_id cpPlatform; // OpenCL platform
  cl_device_id device_id; // device ID
						  // Bind to platform
  clerr = clGetPlatformIDs(1, &cpPlatform, NULL);
  // Get ID for the device
  clerr = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
  // Create a context
  cl_context clctx = clCreateContext(0, 1, &device_id, NULL, NULL, &clerr);
  // Create a command queue
  cl_command_queue clcmdq = clCreateCommandQueue(clctx, device_id, 0, &clerr);


  //@@ Initialize the workgroup dimensions
  int VECTOR_SIZE = inputLength;
  size_t size = VECTOR_SIZE * sizeof(float);
  deviceInput1 = clCreateBuffer(clctx, CL_MEM_READ_ONLY, size, NULL, NULL);
  deviceInput2 = clCreateBuffer(clctx, CL_MEM_READ_ONLY, size, NULL, NULL);
  deviceOutput = clCreateBuffer(clctx, CL_MEM_WRITE_ONLY, size, NULL, NULL);

  clerr = clEnqueueWriteBuffer(clcmdq, deviceInput1, CL_TRUE, 0, size, hostInput1, 0, 0, NULL);
  clerr = clEnqueueWriteBuffer(clcmdq, deviceInput2, CL_TRUE, 0, size, hostInput2, 0, 0, NULL);

  const char* vaddsrc =
	  "__kernel void vadd(__global float *d_A, __global float *d_B, __global float *d_C, int N)   \n" \
       "{                                                                         \n" \
	   "  int id = get_global_id(0);                                              \n" \
	   "   if(id<N)                                                                \n" \
       "{                                                                          \n" \
	   "     d_C[id] = d_A[id] + d_B[id];                                          \n" \
       "}                                                                          \n" \
       "}                                                                          \n" \
	                                                                                "\n";
	  
	  cl_program clpgm;
	  clpgm = clCreateProgramWithSource(clctx, 1, (const char **)&vaddsrc, NULL, &clerr);

	  clerr = clBuildProgram(clpgm, 0, NULL, NULL, NULL, NULL);

	  //you do not need to set this compiler flag in your assignment,
	  //(pass NULL instead of clcompilerflags)

	  cl_kernel clkern = clCreateKernel(clpgm, "vadd", &clerr);

	  clerr = clSetKernelArg(clkern, 0, sizeof(cl_mem), (void *)&deviceInput1);
	  clerr = clSetKernelArg(clkern, 1, sizeof(cl_mem), (void *)&deviceInput2);
	  clerr = clSetKernelArg(clkern, 2, sizeof(cl_mem), (void *)&deviceOutput);
	  clerr = clSetKernelArg(clkern, 3, sizeof(int), &inputLength);

	  cl_event event = NULL;
	  size_t Bsz = 512;
	  size_t Gsz = ((inputLength - 1) / Bsz + 1)*Bsz;
	  clerr = clEnqueueNDRangeKernel(clcmdq, clkern, 1, NULL, &Gsz, &Bsz, 0, NULL, &event);
	  clerr = clWaitForEvents(1, &event);
	  clFinish(clcmdq);
	  clEnqueueReadBuffer(clcmdq, deviceOutput, CL_TRUE, 0, inputLength * sizeof(float), hostOutput, 0, NULL, NULL);


  wbSolution(args, hostOutput, inputLength);

  // release OpenCL resources
  clReleaseMemObject(deviceInput1);
  clReleaseMemObject(deviceInput2);
  clReleaseMemObject(deviceOutput);
  clReleaseProgram(clpgm);
  clReleaseKernel(clkern);
  clReleaseCommandQueue(clcmdq);
  clReleaseContext(clctx);

  // release host memory
  free(hostInput1);
  free(hostInput2);
  free(hostOutput);

  return 0;
}

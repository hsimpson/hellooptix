### PTX (parallel thread execution) Optix Kernel creation

- read file into string
- JIT compile CU to PTX createProgram: nvrtcCreateProgram
- PTX-size nvrtcGetPTXSize
- get PTX nvrtcGetPTX
- destroy program nvrtcDestroyProgram

BLAS = Basic Linear Algebra Subprograms

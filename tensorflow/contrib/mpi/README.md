TODO :-)


Protocol:
grpc+mpi



MPI_OPTIMAL_PATH=[0,1]

Enable the optimal path. Disabled by default as it requires MPI libraries that are CUDA Aware when using GPUs. When using non-CUDA aware MPI libraries and GPUs you will get segmentation faults.


MPI_DISABLED=[0,1]

Disable the MPI path during runtime (e.g. for performance testing)



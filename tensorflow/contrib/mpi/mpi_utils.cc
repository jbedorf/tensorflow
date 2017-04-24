/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/contrib/mpi/mpi_utils.h"
namespace tensorflow {

void MPIUtils::InitMPI() {
  // Initialize the MPI environment if that hasn't been done
  int flag = 0;
  MPICheck(MPI_Initialized(&flag));
  if (!flag) {
    // MPICheck(MPI_Init_thread(0, 0, MPI_THREAD_SINGLE, &flag));
    MPICheck(MPI_Init(0, 0));
    int procId = 0, nProcs = 1, len = -1;
    char procName[kMaxNameLength];
    MPICheck(MPI_Comm_rank(MPI_COMM_WORLD, &procId));
    MPICheck(MPI_Comm_size(MPI_COMM_WORLD, &nProcs));
    MPICheck(MPI_Get_processor_name(procName, &len));
    fprintf(stderr,
            "MPI Environment initialised. Process id: %d Total processes: %d "
            "|| Hostname: %s \n",
            procId, nProcs, procName);
  }
}

MPIUtils::MPIUtils(const std::string& worker_name) {
  InitMPI();
  // Connect the MPI process IDs to the worker names that are used by TF.
  // Gather the names of all the active processes (name can't be longer than
  // 128 bytes)
  int procId = 0, nProcs = 1;
  MPICheck(MPI_Comm_rank(MPI_COMM_WORLD, &procId));
  MPICheck(MPI_Comm_size(MPI_COMM_WORLD, &nProcs));

  char myName[kMaxNameLength];
  CHECK(worker_name.size() < kMaxNameLength)
      << "Specified worker name is too long.";
  snprintf(myName, kMaxNameLength, worker_name.c_str());
  std::vector<char> worker_names(nProcs * kMaxNameLength);
  MPICheck(MPI_Allgather(myName, kMaxNameLength, MPI_CHAR, &worker_names[0],
                         kMaxNameLength, MPI_CHAR, MPI_COMM_WORLD));

  if (procId == 0) LOG(INFO) << "MPI process-ID to gRPC server name map: \n";
  for (int i = 0; i < nProcs; i++) {
    name2id[std::string(&worker_names[i * 128])] = i;
    if (procId == 0)
      LOG(INFO) << "Process: " << i
                << "\tgRPC-name: " << std::string(&worker_names[i * 128])
                << std::endl;
  }
}

}  // namespace tensorflow

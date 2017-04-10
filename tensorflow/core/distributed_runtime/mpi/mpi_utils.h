#ifndef TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_MPI_UTILS_H_
#define TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_MPI_UTILS_H_

#include <string>
#include "tensorflow/core/distributed_runtime/worker_env.h"

#include "third_party/mpi/mpi.h"
#define MPICheck(cmd)                                                 \
  do {                                                                \
    int mpi_errno = cmd;                                              \
    if (MPI_SUCCESS != mpi_errno) {                                   \
      fprintf(stderr, "[%s:%d] MPI call failed with %d \n", __FILE__, \
              __LINE__, mpi_errno);                                   \
      exit(EXIT_FAILURE);                                             \
    }                                                                 \
    assert(MPI_SUCCESS == mpi_errno);                                 \
  } while (false)

namespace tensorflow {
class MPIUtils {

 private:
  // Returns the name of the desitnation specified in a rendezvous key
  // For idx=0 it is the source, for idx=2 it is the destination
  std::string getWorkerName(const std::string& key, const int idx) const {
    // Convert the key back to the subpieces
    const std::vector<string> num_strings = str_util::Split(key, ';');
    // Sanity check, should be 5 src;id;dst;name;frame_iter
    assert(num_strings.size() == 5);
    // Strip the device eg /cpu:0 to get the worker name
    return num_strings[idx].substr(0, num_strings[idx].find_last_of('/'));
  }
  std::map<std::string, int> name2id;

 public:
  MPIUtils(const WorkerEnv* env) {
    // Connect the MPI process IDs to the worker names that are used by TF
    // Gather the names of all the active processes (name can't be longer than
    // 128 bytes)
    int procId = 0, nProcs = 1;
    MPICheck(MPI_Comm_rank(MPI_COMM_WORLD, &procId));
    MPICheck(MPI_Comm_size(MPI_COMM_WORLD, &nProcs));

    const int maxNameLength = 128;
    char myName[maxNameLength];
    CHECK(env->worker_name.size() < maxNameLength)
        << "Specified worker name is too long.";
    strcpy(myName, env->worker_name.c_str());
    std::vector<char> worker_names(nProcs * maxNameLength);
    MPICheck(MPI_Allgather(myName, maxNameLength, MPI_CHAR, &worker_names[0],
                           maxNameLength, MPI_CHAR, MPI_COMM_WORLD));

    if (procId == 0) LOG(INFO) << "MPI process-ID to gRPC server name map: \n";
    for (int i = 0; i < nProcs; i++) {
      name2id[string(&worker_names[i * 128])] = i;
      if (procId == 0)
        LOG(INFO) << "Process: " << i
                  << "\tgRPC-name: " << string(&worker_names[i * 128])
                  << std::endl;
    }
  }

  const int getSourceID(const std::string& key) const {
    auto it = name2id.find(getWorkerName(key, 0));
    if (it == name2id.end()) {
      LOG(FATAL) << "Failed to convert worker name to MPI index: " << key;
    }
    return it->second;
  }
};
}  // namespace tensorflow

#endif

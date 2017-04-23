#ifndef TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_MPI_UTILS_H_
#define TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_MPI_UTILS_H_

#include <string>
#include <map>
#include <vector>

#include "tensorflow/core/lib/strings/str_util.h"

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
  const uint kMaxNameLength = 128;
  std::map<std::string, int> name2id;

  // Returns the name of the destination specified in a rendezvous key
  // For idx=0 it is the source, for idx=2 it is the destination
  std::string getWorkerName(const std::string& key, const int idx) const {
    const std::vector<std::string> num_strings = str_util::Split(key, ';');
    // Sanity check, should be 5 src;id;dst;name;frame_iter
    assert(num_strings.size() == 5);
    // Strip the device eg /cpu:0 to get the worker name
    return num_strings[idx].substr(0, num_strings[idx].find_last_of('/'));
  }

  void InitMPI();

 public:
  explicit MPIUtils(const std::string& worker_name);

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

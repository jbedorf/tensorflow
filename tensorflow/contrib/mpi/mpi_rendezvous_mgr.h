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

#ifndef TENSORFLOW_CONTRIB_MPI_MPI_RENDEZVOUS_MGR_H_
#define TENSORFLOW_CONTRIB_MPI_MPI_RENDEZVOUS_MGR_H_

#include <queue>
#include <thread>
#include <list>
#include <string>
#include <memory>
#include <map>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tensorflow/contrib/mpi/mpi_utils.h"
#include "tensorflow/core/distributed_runtime/base_rendezvous_mgr.h"
#include "tensorflow/core/distributed_runtime/worker_env.h"
#include "tensorflow/contrib/mpi/mpi_msg.pb.h"
#include "tensorflow/core/protobuf/worker.pb.h"

#define TAG_REQTENSOR 1010
#define TAG_SENDTENSOR 2020
#define TAG_SENDTENSOR2 3030

namespace tensorflow {

class MPISendTensorCall {
 public:
  char* sendBuff;
  char* sendBuff2;

  MPI_Request msg1;
  MPI_Request msg2;
  int done1;  // Int instead of bool for simpler isFinished logic
  int done2;
  MPIRecvTensorResponse mRes;
  Notification n;

 public:
  MPISendTensorCall()
      : sendBuff(nullptr), sendBuff2(nullptr), done1(0), done2(1) {}

  ~MPISendTensorCall() {
    n.Notify();
    delete[] sendBuff;
    delete[] sendBuff2;
  }

  MPISendTensorCall(MPISendTensorCall&&) = delete;

  void Init(const Rendezvous::ParsedKey& parsed, const int64 step_id,
            const bool is_dead) {
    mRes.set_key(parsed.FullKey().ToString());
    mRes.set_step_id(step_id);
    mRes.mutable_response()->set_is_dead(is_dead);
    mRes.mutable_response()->set_send_start_micros(Env::Default()->NowMicros());
    mRes.set_singlesend(true);
  }

  bool isFinished() {
    MPI_Status status;
    if (!done1) MPICheck(MPI_Test(&msg1, &done1, &status));
    if (!done2) MPICheck(MPI_Test(&msg2, &done2, &status));
    return done1 && done2;
  }
};

class MPIRendezvousCall {
 public:
  Rendezvous::DoneCallback done_;
  RecvTensorRequest req_;
  MPI_Request mpiReq;
  char* reqBuff;
  size_t reqBuffSize;
  std::function<void(MPIRecvTensorResponse)> recvCB;

  MPIRendezvousCall() : reqBuff(nullptr) {}
  ~MPIRendezvousCall() {
    MPICheck(MPI_Wait(&mpiReq, MPI_STATUS_IGNORE));
    delete[] reqBuff;
  }

  void Init(const Rendezvous::ParsedKey& parsed, const int64 step_id) {
    req_.set_step_id(step_id);
    req_.set_rendezvous_key(parsed.FullKey().data(), parsed.FullKey().size());
    reqBuffSize = req_.ByteSize();
    reqBuff = new char[reqBuffSize];
    req_.SerializeToArray(reqBuff, reqBuffSize);
  }
};

class MPIRemoteRendezvous : public BaseRemoteRendezvous {
 public:
  MPIRemoteRendezvous(const WorkerEnv* env, const string& worker_name,
                      int64 step_id, const MPIUtils* util,
                      BaseRendezvousMgr* mgr_)
      : BaseRemoteRendezvous(env, worker_name, step_id, false),
        mpiUtils_(util),
        rendezvous_mgr(mgr_) {}

 protected:
  void RecvFromRemoteAsync(const Rendezvous::ParsedKey& parsed,
                           const Rendezvous::Args& args,
                           DoneCallback done) override;

 private:
  ~MPIRemoteRendezvous() override;

  const MPIUtils* mpiUtils_;
  BaseRendezvousMgr* rendezvous_mgr;

  TF_DISALLOW_COPY_AND_ASSIGN(MPIRemoteRendezvous);
};

class MPIRendezvousMgr : public BaseRendezvousMgr {
 public:
  explicit MPIRendezvousMgr(const WorkerEnv* env, const string& worker_name,
                            WorkerCacheInterface* worker_cache);
  ~MPIRendezvousMgr() {
    delete mpiUtils_;
    fprintf(stderr, "Delete MPIRendezvousMgr \n");

    // TODO(jbedorf) stop requestThread

    MPICheck(MPI_Finalize());
  }

 protected:
  BaseRemoteRendezvous* Create(int64 step_id, const WorkerEnv* worker_env,
                               const string& session_name) override;

  const WorkerEnv* worker_env_2;

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(MPIRendezvousMgr);

  typedef std::function<MPISendTensorCall*(
      const Status&, const Rendezvous::Args&, const Rendezvous::Args&,
      const Tensor&, const bool, MPISendTensorCall*)> MPIRecvTensorCallBack;

  typedef std::pair<std::string, std::function<void()>> requestQueueEntry;
  typedef std::pair<std::string, std::function<MPISendTensorCall*()>>
      sendQueueEntry;

  mutex msq_;
  std::queue<sendQueueEntry> sendQueue GUARDED_BY(msq_);
  mutex mrq_;
  std::queue<requestQueueEntry> requestQueue GUARDED_BY(mrq_);
  std::map<int64,
           std::unordered_map<std::string, std::shared_ptr<MPIRendezvousCall>>>
      recvTensorList GUARDED_BY(mrq_);
  void addRequest(RecvTensorRequest, const int);

  std::thread requestThread;
  void MPIBackgroundThread();

  MPIUtils* mpiUtils_;

  bool doOptimalPath;

 public:
  void queueRequest(std::string key, int64 step_id,
                    std::function<void()> reqTensor, MPIRendezvousCall* rCall) {
    mutex_lock l(mrq_);
    requestQueue.push(requestQueueEntry(key, std::move(reqTensor)));
    recvTensorList[step_id][key] = std::shared_ptr<MPIRendezvousCall>(rCall);
  }

  void queueSendRequest(sendQueueEntry req) {
    mutex_lock l(msq_);
    sendQueue.push(req);
  }

  bool getRecvCall(const int64 step_id, const std::string& key,
                   std::shared_ptr<MPIRendezvousCall>* call) {
    mutex_lock l(mrq_);
    if (recvTensorList.find(step_id) == recvTensorList.end()) {
      LOG(FATAL) << "Step not found in recvTensorList, step: " << step_id;
      abort();
    }
    if (recvTensorList[step_id].find(key) != recvTensorList[step_id].end()) {
      *call = recvTensorList[step_id][key];
    } else {
      LOG(FATAL) << "Key not found in recvTensorList, key: " << key;
      abort();
    }
  }

  void removeRecvCall(const int64 step_id, const std::string& key) {
    mutex_lock l(mrq_);
    recvTensorList[step_id].erase(key);
  }

  void removeStepID(const int64 step_id) {
    mutex_lock l(mrq_);
    recvTensorList.erase(step_id);
    // TODO(jbedorf) Should we verify that the step_id is clear before remove?
  }

  bool getRequest(requestQueueEntry* req) {
    mutex_lock l(mrq_);
    if (!requestQueue.empty()) {
      *req = requestQueue.front();
      requestQueue.pop();
      return true;
    }
    return false;
  }

  bool getResponse(sendQueueEntry* send) {
    mutex_lock l(msq_);
    if (!sendQueue.empty()) {
      *send = sendQueue.front();
      sendQueue.pop();
      return true;
    }
    return false;
  }

  template <typename T>
  int probeForData(const int tag, MPI_Status* status, T* obj) {
    int flag = 0, incSize = 0;
    MPI_Message msg;

    // Receive the message, probe as size is variable
    MPICheck(
        MPI_Improbe(MPI_ANY_SOURCE, tag, MPI_COMM_WORLD, &flag, &msg, status));
    if (flag) {
      MPICheck(MPI_Get_count(status, MPI_CHAR, &incSize));
      std::vector<char> reqBuff(incSize);
      MPICheck(MPI_Mrecv(&reqBuff[0], incSize, MPI_CHAR, &msg, status));
      obj->ParseFromArray(&reqBuff[0], reqBuff.size());
    }
    return flag;
  }
};  // MPIRendezvousMgr
}  // namespace tensorflow

#endif  // TENSORFLOW_CONTRIB_MPI_MPI_RENDEZVOUS_MGR_H_

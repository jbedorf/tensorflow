
#ifndef TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_MPI_RENDEZVOUS_MGR_H_
#define TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_MPI_RENDEZVOUS_MGR_H_

#include <queue>
#include <thread>
#include <list>

#include "tensorflow/core/distributed_runtime/mpi/mpi_utils.h"
#include "tensorflow/core/distributed_runtime/base_rendezvous_mgr.h"
#include "tensorflow/core/distributed_runtime/worker_env.h"
#include "tensorflow/core/distributed_runtime/mpi/mpimsg.pb.h"
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
  MPIRecvTensorRequest req_;
  MPI_Request mpiReq;
  char* reqBuff;
  size_t reqBuffSize;
  std::function<void(MPIRecvTensorResponse)> recvCB;

  MPIRendezvousCall() : reqBuff(nullptr) {}
  ~MPIRendezvousCall() {
    MPI_Status status;
    MPICheck(MPI_Wait(&mpiReq, &status));
    delete[] reqBuff;
  }

  void Init(const Rendezvous::ParsedKey& parsed, const int64 step_id) {
    req_.mutable_request()->set_step_id(step_id);
    req_.mutable_request()->set_rendezvous_key(parsed.FullKey().data(),
                                               parsed.FullKey().size());
    reqBuffSize = req_.ByteSize();
    reqBuff = new char[reqBuffSize];
    req_.SerializeToArray(reqBuff, reqBuffSize);
  }
};

class MPIRemoteRendezvous : public BaseRemoteRendezvous {

 protected:
  void RecvFromRemoteAsync(const Rendezvous::ParsedKey& parsed,
                           const Rendezvous::Args& args,
                           DoneCallback done) override;
  void SendToRemote(const Rendezvous::ParsedKey& key,
                    const Rendezvous::Args& args, const Tensor& val,
                    const bool is_dead, Status& s) {
    // TODO delete, when basing on latest master
  }

 private:
  ~MPIRemoteRendezvous() override {
    // TODO Clean up anything left from this step
  }

  const MPIUtils* mpiUtils_;

  TF_DISALLOW_COPY_AND_ASSIGN(MPIRemoteRendezvous);

 public:
  MPIRemoteRendezvous(const WorkerEnv* env, int64 step_id, const MPIUtils* util)
      : BaseRemoteRendezvous(env, step_id, true), mpiUtils_(util) {}
};

class MPIRendezvousMgr : public BaseRendezvousMgr {
 public:
  explicit MPIRendezvousMgr(const WorkerEnv* env);
  ~MPIRendezvousMgr() {
    delete mpiUtils_;
    fprintf(stderr, "DELETE MANAGER \n");

    // TODO stop requestThread

    MPICheck(MPI_Finalize());
  }

 protected:
  BaseRemoteRendezvous* Create(int64 step_id,
                               const WorkerEnv* worker_env) override;

  const WorkerEnv* worker_env_2;

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(MPIRendezvousMgr);

  typedef std::function<MPISendTensorCall*(
      const Status&, const Rendezvous::Args&, const Rendezvous::Args&,
      const Tensor&, const bool, MPISendTensorCall*)> MPIRecvTensorCallBack;

  void addRequest(MPIRecvTensorRequest, const int);

  MPIUtils* mpiUtils_;

  std::thread requestThread;

  mutex msq_;
  mutex mrq_;
  std::queue<std::pair<std::string, std::function<MPISendTensorCall*()>>>
      sendQueue GUARDED_BY(msq_);
  std::queue<std::pair<std::string, std::function<void()>>> requestQueue
      GUARDED_BY(mrq_);
  std::map<int64,
           std::unordered_map<std::string, std::shared_ptr<MPIRendezvousCall>>>
      recvTensorList GUARDED_BY(mrq_);

  void runThread() {
    std::list<std::unique_ptr<MPISendTensorCall>> runningSends;

    while (1) {

      MPI_Message msg;
      MPI_Status status;
      int flag = 0, incSize = 0;
      // Check for incoming Tensor requests
      {
        // Receive the header message, probe as size is variable
        MPICheck(MPI_Improbe(MPI_ANY_SOURCE, TAG_REQTENSOR, MPI_COMM_WORLD,
                             &flag, &msg, &status));
        if (flag) {
          MPICheck(MPI_Get_count(&status, MPI_CHAR, &incSize));
          std::vector<char> reqBuff(incSize);
          MPICheck(MPI_Mrecv(&reqBuff[0], incSize, MPI_CHAR, &msg, &status));

          MPIRecvTensorRequest mReq;
          mReq.ParseFromArray(&reqBuff[0], reqBuff.size());
          this->addRequest(mReq, status.MPI_SOURCE);
        }  // flag
      }    // section

      // Check for incoming Tensor reply
      {
        // Receive the Tensor reply message, probe as size is variable
        MPICheck(MPI_Improbe(MPI_ANY_SOURCE, TAG_SENDTENSOR, MPI_COMM_WORLD,
                             &flag, &msg, &status));
        if (flag) {
          MPICheck(MPI_Get_count(&status, MPI_CHAR, &incSize));
          std::vector<char> resBuff(incSize);
          MPICheck(MPI_Mrecv(&resBuff[0], incSize, MPI_CHAR, &msg, &status));

          MPIRecvTensorResponse mRes;
          mRes.ParseFromArray(&resBuff[0], resBuff.size());
          const int64 step_id = mRes.step_id();
          std::string key = mRes.key();

          std::shared_ptr<MPIRendezvousCall> call;
          mrq_.lock();
          if (recvTensorList.find(step_id) == recvTensorList.end()) {
            LOG(FATAL) << "Step not found in recvTensorList, step: " << step_id;
            abort();
          }
          if (recvTensorList[step_id].find(key) !=
              recvTensorList[step_id].end())
            call = recvTensorList[step_id][key];
          else {
            LOG(FATAL) << "Key not found in recvTensorList, key: " << key;
            abort();
          }
          mrq_.unlock();

          call->recvCB(mRes);

          mrq_.lock();
          recvTensorList[step_id].erase(key);
          mrq_.unlock();
        }
      }

      // Remove sends that have been completed
      runningSends.remove_if([](std::unique_ptr<MPISendTensorCall>& i) {
        return i->isFinished();
      });

      // send a Tensor request
      mrq_.lock();
      if (!requestQueue.empty()) {
        auto x = requestQueue.front();
        requestQueue.pop();
        mrq_.unlock();
        x.second();
      } else {
        mrq_.unlock();
      }

      // Send a Tensor response
      msq_.lock();
      if (!sendQueue.empty()) {
        auto x = sendQueue.front();
        sendQueue.pop();
        msq_.unlock();
        std::unique_ptr<MPISendTensorCall> p(x.second());
        runningSends.push_back(std::move(p));
      } else {
        msq_.unlock();
      }

      // std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }

 public:
  void queueRequest(std::string key, int64 step_id,
                    std::function<void()> reqTensor, MPIRendezvousCall* rCall) {
    mrq_.lock();
    std::pair<std::string, std::function<void()>> req(key,
                                                      std::move(reqTensor));
    requestQueue.push(std::move(req));
    recvTensorList[step_id][key] = std::shared_ptr<MPIRendezvousCall>(rCall);
    mrq_.unlock();
  }

  void queueSendRequest(
      std::pair<std::string, std::function<MPISendTensorCall*()>> req) {
    msq_.lock();
    sendQueue.push(req);
    msq_.unlock();
  }

};  // MPIRendezvousMgr
}  // namespace

#endif

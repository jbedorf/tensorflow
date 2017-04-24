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

#include "tensorflow/contrib/mpi/mpi_rendezvous_mgr.h"

#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/core/distributed_runtime/tensor_coding.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/gpu/gpu_util.h"
#include "tensorflow/core/distributed_runtime/session_mgr.h"

namespace tensorflow {

/*
 * Add the request for one of our Tensors by a remote process
 * to the local send/table. The here created callback will
 * be called once the Tensor data has arrived and is
 * ready to be send to the remote requester.
 */
void MPIRendezvousMgr::addRequest(RecvTensorRequest mReq,
                                  const int mpi_dst) {
  const int64 step_id = mReq.step_id();
  const std::string& key = mReq.rendezvous_key();
  Rendezvous::ParsedKey parsed;
  Status s = Rendezvous::ParseKey(key, &parsed);

  MPIRecvTensorCallBack cb = [this, mpi_dst](
      const Status& status, const Rendezvous::Args& send_args,
      const Rendezvous::Args& recv_args, const Tensor& val, bool is_dead,
      MPISendTensorCall* mpiSC) {
    // TODO(jbedorf) this should be a loop over max size
    MPICheck(MPI_Isend(mpiSC->sendBuff,
                       static_cast<int>(mpiSC->mRes.ByteSize()), MPI_CHAR,
                       mpi_dst, TAG_SENDTENSOR, MPI_COMM_WORLD, &mpiSC->msg1));
    mpiSC->done1 = 0;

    if (!mpiSC->mRes.singlesend()) {
      const size_t nBytes = val.TotalBytes();
      void* temp = const_cast<void*>(DMAHelper::base(&val));

      // If the MPI environment is not GPU aware there should be a data transfer
      // here
      // if(src_dev->tensorflow_gpu_device_info()) //memcpy to sendBuff2

      // TODO(jbedorf)  this should be a loop over max size
      MPICheck(MPI_Isend(temp, static_cast<int>(nBytes), MPI_CHAR, mpi_dst,
                         TAG_SENDTENSOR2, MPI_COMM_WORLD, &mpiSC->msg2));
      mpiSC->done2 = 0;
    }
    return mpiSC;
  };

  // Wrapper around the read callback to place the callback on our queue
  Rendezvous::DoneCallback cb2 = [this, parsed, step_id, cb](
      const Status& status, const Rendezvous::Args& send_args,
      const Rendezvous::Args& recv_args, const Tensor& val, bool is_dead) {
    if (!status.ok()) {
      std::cerr << "RecvLocal was not ok: " << parsed.FullKey()
                << "  error: " << status.error_message() << std::endl;
      abort();
      return;
    }

    VLOG(3) << "MPI Sending tensor " << parsed.FullKey()
            << " @ step: " << step_id << std::endl;

    auto mpiSC = new MPISendTensorCall();
    mpiSC->Init(parsed, step_id, is_dead);

    Device* src_dev = nullptr;
    Status s = this->worker_env_2->device_mgr->LookupDevice(parsed.src_device,
                                                            &src_dev);
    CHECK(s.ok()) << "src device not found";

    // Control if shape and data should be send together or if we can optimize
    // it in two different transfers, thereby reducing memory copies
    bool doOptimalTransfer = true;
    if (!DataTypeCanUseMemcpy(val.dtype())) doOptimalTransfer = false;
    if (val.TotalBytes() < 1024) doOptimalTransfer = false;

    doOptimalTransfer = doOptimalTransfer && doOptimalPath;

    if (doOptimalTransfer) {
      // First send the Tensor description and in a follow up transfer the data
      mpiSC->mRes.mutable_response()->mutable_tensor()->set_dtype(val.dtype());
      val.shape().AsProto(mpiSC->mRes.mutable_response()
                              ->mutable_tensor()
                              ->mutable_tensor_shape());
      mpiSC->mRes.set_singlesend(false);
    } else {
      // Send the Tensor description and data in a single transfer
      if (src_dev->tensorflow_gpu_device_info() &&
          (!send_args.alloc_attrs.on_host())) {
        Notification n;
        GPUUtil::SetProtoFromGPU(
            val, src_dev, send_args.device_context,
            mpiSC->mRes.mutable_response()->mutable_tensor(), is_dead,
            [&n, &s](const Status& s_) {
              s = s_;
              n.Notify();
            });
        n.WaitForNotification();
      } else {
        val.AsProtoTensorContent(
            mpiSC->mRes.mutable_response()->mutable_tensor());
      }
    }

    mpiSC->sendBuff = new char[mpiSC->mRes.ByteSize()];
    mpiSC->mRes.SerializeToArray(mpiSC->sendBuff, mpiSC->mRes.ByteSize());

    std::function<MPISendTensorCall*()> res =
        std::bind(cb, status, send_args, recv_args, val, is_dead, mpiSC);

    sendQueueEntry req(parsed.FullKey().ToString().c_str(), std::move(res));

    this->queueSendRequest(req);

    // Wait for the notification that indicates the tensor has been
    // succesfully transmitted to the remote process. Only needed if we
    // have not parsed the tensor to proto
    if (doOptimalTransfer) mpiSC->n.WaitForNotification();
  };  // cb2

  worker_env_2->compute_pool->Schedule([this, step_id, parsed, cb2]() {
    this->RecvLocalAsync(step_id, parsed, cb2);
  });
}

MPIRendezvousMgr::MPIRendezvousMgr(const WorkerEnv* env,
                                   const string& worker_name,
                                   WorkerCacheInterface* worker_cache)
    : BaseRendezvousMgr(env, worker_name),
      worker_env_2(env),
      doOptimalPath(false) {

  const char* mpienv = getenv("MPI_OPTIMAL_PATH");
  if (mpienv && mpienv[0] == '1') {
    LOG(INFO) << "MPI Optimal copy path enabled (Requires CUDA-Aware MPI when "
                 "using GPUs)\n";
    doOptimalPath = true;
  }

  mpiUtils_ = new MPIUtils(worker_name);
  requestThread = std::thread(&MPIRendezvousMgr::MPIBackgroundThread, this);
}

BaseRemoteRendezvous* MPIRendezvousMgr::Create(int64 step_id,
                                               const WorkerEnv* worker_env,
                                               const string& worker_name) {
  return new MPIRemoteRendezvous(worker_env, worker_name, step_id, mpiUtils_,
                                 this);
}

void MPIRendezvousMgr::MPIBackgroundThread() {
  std::list<std::unique_ptr<MPISendTensorCall>> runningSends;

  while (1) {
    MPI_Status status;

    // Check for incoming Tensor requests
    RecvTensorRequest mReq;
    if (probeForData(TAG_REQTENSOR, &status, &mReq)) {
      this->addRequest(mReq, status.MPI_SOURCE);
    }

    // Check for incoming Tensor reply
    MPIRecvTensorResponse mRes;
    if (probeForData(TAG_SENDTENSOR, &status, &mRes)) {
      const int64 step_id = mRes.step_id();
      std::string key = mRes.key();

      std::shared_ptr<MPIRendezvousCall> call;
      getRecvCall(step_id, key, &call);
      call->recvCB(mRes);
      removeRecvCall(step_id, key);
    }

    // Remove sends that have been completed
    runningSends.remove_if([](std::unique_ptr<MPISendTensorCall>& i) {
      return i->isFinished();
    });

    // send a Tensor request
    requestQueueEntry req;
    if (getRequest(&req)) req.second();

    // Send a Tensor response
    sendQueueEntry send;
    if (getResponse(&send)) {
      std::unique_ptr<MPISendTensorCall> p(send.second());
      runningSends.push_back(std::move(p));
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

void MPIRemoteRendezvous::RecvFromRemoteAsync(
    const Rendezvous::ParsedKey& parsed, const Rendezvous::Args& recv_args,
    DoneCallback done) {

  Status s = Status::OK();
  MPIRendezvousCall* reqCall = new MPIRendezvousCall();

  VLOG(2) << "MPI User requested " << parsed.FullKey()
          << " @ step: " << step_id_ << std::endl;

  const int dst = mpiUtils_->getSourceID(parsed.FullKey().ToString());

  Device* dst_device;
  if (s.ok()) {
    s = env_->device_mgr->LookupDevice(parsed.dst_device, &dst_device);
  }
  if (!s.ok()) {
    done(s, Args(), recv_args, Tensor{}, false);
    return;
  }

  // Set properties of the request object and create the request function
  reqCall->Init(parsed, step_id_);

  std::function<void()> reqTensor = [parsed, dst, reqCall]() {
    MPICheck(MPI_Isend(reqCall->reqBuff, reqCall->reqBuffSize, MPI_CHAR, dst,
                       TAG_REQTENSOR, MPI_COMM_WORLD, &reqCall->mpiReq));
  };

  // Create the function which is called when the Tensor is send by remote
  const int64 temp1 = step_id_;
  reqCall->recvCB = [this, parsed, recv_args, done, dst, temp1, reqCall](
      MPIRecvTensorResponse mRes) {
    Status s;
    Device* dst_device;
    if (s.ok()) {
      s = env_->device_mgr->LookupDevice(parsed.dst_device, &dst_device);
    }

    VLOG(3) << "MPI Received tensor " << parsed.FullKey()
            << " @ step: " << temp1 << " single-send: " << mRes.singlesend()
            << std::endl;

    Tensor val;
    if (mRes.singlesend()) {
      dst_device->MakeTensorFromProto(mRes.response().tensor(),
                                      recv_args.alloc_attrs, &val);
    } else {
      TensorResponse tr;
      tr.InitAlloc(dst_device, recv_args.alloc_attrs);
      tr.InitPartial(mRes.response());
      const size_t nBytes = tr.tensor().TotalBytes();
      void* data = const_cast<void*>(DMAHelper::base(&tr.tensor()));
      MPI_Status status;
      MPICheck(MPI_Recv(data, static_cast<int>(nBytes), MPI_BYTE, dst,
                        TAG_SENDTENSOR2, MPI_COMM_WORLD, &status));
      val = std::move(tr.tensor());
    }

    done(s, Args(), recv_args, val, mRes.response().is_dead());
  };

  MPIRendezvousMgr* mgr = dynamic_cast<MPIRendezvousMgr*>(this->rendezvous_mgr);
  mgr->queueRequest(parsed.FullKey().ToString(), step_id_, std::move(reqTensor),
                    reqCall);
}

MPIRemoteRendezvous::~MPIRemoteRendezvous() {
  MPIRendezvousMgr* mgr = dynamic_cast<MPIRendezvousMgr*>(this->rendezvous_mgr);
  mgr->removeStepID(step_id_);
}

}  // namespace tensorflow

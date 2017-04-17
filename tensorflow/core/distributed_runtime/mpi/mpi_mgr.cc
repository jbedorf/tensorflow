
#include "tensorflow/core/distributed_runtime/tensor_coding.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/gpu/gpu_util.h"

#include <chrono>
#include <functional>
#include <memory>

#include "tensorflow/core/distributed_runtime/mpi/mpi_mgr.h"

namespace tensorflow {

/*
 * Add the request for one of our Tensors by a remote process
 * to the local send/table. The here created callback will
 * be called once the Tensor data has arrived and is
 * ready to be send to the remote requester.
 */
void MPIRendezvousMgr::addRequest(MPIRecvTensorRequest mReq,
                                  const int mpi_dst) {
  const int64 step_id = mReq.request().step_id();
  const string& key = mReq.request().rendezvous_key();
  const int followUpTag = mReq.followuptag();
  Rendezvous::ParsedKey parsed;
  Status s = Rendezvous::ParseKey(key, &parsed);

  MPIRecvTensorCallBack cb = [this, key, step_id, parsed, mpi_dst, followUpTag](
      const Status& status, const Rendezvous::Args& send_args,
      const Rendezvous::Args& recv_args, const Tensor& val, bool is_dead,
      MPISendTensorCall* mpiSC) {

    // TODO this should be a loop over max size
    MPICheck(MPI_Isend(mpiSC->sendBuff, (int)mpiSC->mRes.ByteSize(), MPI_CHAR,
                       mpi_dst, TAG_SENDTENSOR, MPI_COMM_WORLD, &mpiSC->msg1));
    mpiSC->done1 = 0;

    if (!mpiSC->mRes.singlesend()) {
      const size_t nBytes = val.TotalBytes();
      void* temp = const_cast<void*>(DMAHelper::base(&val));

      // If the MPI environment is not GPU aware there should be a data transfer
      // here
      // if(src_dev->tensorflow_gpu_device_info()) //memcpy to sendBuff2

      // TODO this should be a loop over max size
      MPICheck(MPI_Isend(temp, (int)nBytes, MPI_CHAR, mpi_dst, TAG_SENDTENSOR2,
                         MPI_COMM_WORLD, &mpiSC->msg2));
      mpiSC->done2 = 0;
    }
    return mpiSC;
  };

  using namespace std::placeholders;
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
    // it
    // in two different transfers, thereby reducing memory copies
    bool doOptimalTransfer = true;
    if (!DataTypeCanUseMemcpy(val.dtype())) doOptimalTransfer = false;
    if (val.TotalBytes() < 1024) doOptimalTransfer = false;

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

    std::pair<std::string, std::function<MPISendTensorCall*()> > req(
        parsed.FullKey().ToString().c_str(), std::move(res));

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

void MPIRemoteRendezvous::RecvFromRemoteAsync(
    const Rendezvous::ParsedKey& parsed, const Rendezvous::Args& recv_args,
    DoneCallback done) {

  Status s = Status::OK();
  MPIRendezvousCall* reqCall = new MPIRendezvousCall();

  VLOG(3) << "MPI User requested " << parsed.FullKey()
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
      MPICheck(MPI_Recv(data, (int)nBytes, MPI_BYTE, dst, TAG_SENDTENSOR2,
                        MPI_COMM_WORLD, &status));
      val = std::move(tr.tensor());
    }

    done(s, Args(), recv_args, val, mRes.response().is_dead());
  };

  MPIRendezvousMgr* mgr = dynamic_cast<MPIRendezvousMgr*>(env_->rendezvous_mgr);
  mgr->queueRequest(parsed.FullKey().ToString(), step_id_, std::move(reqTensor),
                    reqCall);
}

MPIRendezvousMgr::MPIRendezvousMgr(const WorkerEnv* env)
    : BaseRendezvousMgr(env), worker_env_2(env) {

  mpiUtils_ = new MPIUtils(env);
  requestThread = std::thread(&MPIRendezvousMgr::runThread, this);
}

BaseRemoteRendezvous* MPIRendezvousMgr::Create(int64 step_id,
                                               const WorkerEnv* worker_env) {
  return new MPIRemoteRendezvous(worker_env, step_id, mpiUtils_);
}

}  // end namespace

"""Benchmark tensorflow distributed by adding vector of ones on worker2
to variable on worker1 as fast as possible.

On 2014 macbook, TensorFlow 0.10 this shows

Local rate:       2175.28 MB per second
Distributed rate: 107.13 MB per second

"""

import subprocess
import tensorflow as tf
import time
import sys
import os
import socket
from tensorflow.python.client import device_lib


flags = tf.flags
flags.DEFINE_integer("iters", 10, "Maximum number of additions")
flags.DEFINE_integer("data_mb", 100, "size of vector in MBs")
flags.DEFINE_string("port1", "12222", "port of worker1")
flags.DEFINE_string("port2", "12220", "port of worker2")
flags.DEFINE_string("task", "", "internal use")
FLAGS = flags.FLAGS



# setup local cluster from flags
#host = "127.0.0.1:"
#cluster = {"worker": [host+FLAGS.port1, host+FLAGS.port2]}
#cluster = {"worker": [ "cs1:12222"] , "master" : ["cs0:12220"]}


cluster = {"worker": [ "localhost:12222"] , "master" : ["localhost:12220"]}

#cluster = {"worker": [ "10.1.1.51:12222"] , "master" : ["10.1.1.50:12220"]}


#cluster = {"worker": [ "p10a117:12222"] , "master" : ["p10a114:12220"]}
#cluster = {"worker": [ "192.168.0.11:12222" ] , "master": [ "192.168.0.12:12220" ] } #cs1 and cs2
#cluster = {"worker": [ "cs1:12222"] , "master" : ["cs1:12220"]}
clusterspec = tf.train.ClusterSpec(cluster).as_cluster_def()

def default_config():
  optimizer_options = tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L0)
  config = tf.ConfigProto(
    graph_options=tf.GraphOptions(optimizer_options=optimizer_options))
  config.log_device_placement = False
  #config.log_device_placement = True
  config.allow_soft_placement = False
  return config

def create_graph(device1, device2, sizeX):
  """Create graph that keeps variable on device1 and
  vector of ones/addition op on device2"""
  
  tf.reset_default_graph()
  dtype=tf.int32
  params_size = sizeX # 1MB is 250k integers

  with tf.device(device1):
    params  = tf.get_variable("params", [params_size], dtype,
                             initializer=tf.zeros_initializer())

  with tf.device(device2):
    # constant node gets placed on device1 because of simple_placer
    update = tf.get_variable("update", [params_size], dtype,
                             initializer=tf.ones_initializer())
    add_op = params.assign_add(update)

  init_op = tf.initialize_all_variables()
  #return init_op, add_op, temp_op
  return init_op, add_op

#def run_benchmark(sess, init_op, add_op, temp_op):
def run_benchmark(size, sess, init_op, add_op):
  """Returns MB/s rate of addition."""
  
  sess.run(init_op)
  sess.run(add_op.op)  # warm-up
  start_time = time.time()
  for i in range(FLAGS.iters):
    # change to add_op.op to make faster
    sess.run(add_op)
  elapsed_time = time.time() - start_time
  run_metadata = tf.RunMetadata()
  
  sizeMB = size / (1024*1024.)

  #return float(FLAGS.iters)*FLAGS.data_mb/elapsed_time
  return float(FLAGS.iters)*sizeMB/elapsed_time


if __name__=='__main__':
  #procId = int(os.environ['MV2_COMM_WORLD_RANK']) 
  procId = int(os.environ['OMPI_COMM_WORLD_RANK'])
  FLAGS.task = procId
  #tf.logging.set_verbosity(10)
  print("I am rank: %d on host: %s \n" % (procId, socket.gethostname()))
  os.environ['CUDA_VISIBLE_DEVICES'] = ""

  if 1:
      if not FLAGS.task:
        #sess  = tf.Session("grpc://"+host+FLAGS.port1, config=default_config())
        cluster = tf.train.ClusterSpec(clusterspec)
        server  = tf.train.Server(cluster, job_name="worker")

        sizeX = 128
        for i in range(1):
        #for i in range(16):
            ops     = create_graph("/job:worker/task:0/cpu:0", "/job:master/task:0/cpu:0", sizeX)
            #ops     = create_graph("/job:master/task:0/cpu:0","/job:worker/task:0/cpu:0", sizeX)
            sess    = tf.Session(server.target, config=default_config())
            rate2 = run_benchmark(sizeX, sess, *ops)
            rate1 = 0

            print("Adding data in %d MB chunks" %(FLAGS.data_mb))
            print("Local rate:       %.2f MB per second" %(rate1,))
            print("Distributed rate: %.2f MB per second" %(rate2,))

            sizeX *=2 

            sess.reset(server.target)

      else: # Launch TensorFlow server
        print("Launching Server")
        server = tf.train.Server(clusterspec, config=default_config(),
                                 job_name="master",
                                 task_index=0)
        server.join()

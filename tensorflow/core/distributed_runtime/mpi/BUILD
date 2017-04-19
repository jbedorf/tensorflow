# Description:
#   MPI communication interfaces and implementations for TensorFlow.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)

filegroup(
    name = "c_srcs",
    data = glob([
        "**/*.cc",
        "**/*.h",
    ]),
)

load(
    "//tensorflow:tensorflow.bzl",
    "tf_cuda_library",
)
load("//tensorflow:tensorflow.bzl", "tf_cuda_cc_tests")

# For platform specific build config
load(
    "//tensorflow/core:platform/default/build_config.bzl",
    "tf_kernel_tests_linkstatic",
)
load("//tensorflow/core:platform/default/build_config.bzl", "tf_proto_library")
load(
    "//tensorflow/core:platform/default/build_config_root.bzl",
    "tf_cuda_tests_tags",
)

package(default_visibility = [
    "//tensorflow:internal",
])

cc_library(
    name = "mpi_mgr",
    srcs = ["mpi_mgr.cc"],
    hdrs = ["mpi_mgr.h", "mpi_utils.h"],
    deps = [
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:gpu_runtime",
        "//tensorflow/core/distributed_runtime:base_rendezvous_mgr",
        "//tensorflow/core/distributed_runtime:worker_env",
        "//tensorflow/core/distributed_runtime:session_mgr",
        ":mpimsg_proto_cc",
        "//tensorflow/core:protos_cc",
        "//tensorflow/core:worker_proto_cc",
        "//third_party/mpi" 
    ],
)


tf_proto_library(
    name = "mpimsg_proto",
    srcs  = ["mpimsg.proto"],
    cc_api_version = 2,
    protodeps = ["//tensorflow/core:worker_proto"],
    visibility = ["//tensorflow:internal"],
)


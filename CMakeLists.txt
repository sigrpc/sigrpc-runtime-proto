cmake_minimum_required(VERSION 3.8)

project(sigrpc-runtime C CXX)

include(common.cmake)

add_compile_options(-g -O0 -fPIC -mfsgsbase)

# Include generated *.pb.h files
include_directories("${CMAKE_CURRENT_BINARY_DIR}")

# sigrpc_grpc_proto
add_library(sigrpc_grpc_proto
  "${CMAKE_CURRENT_BINARY_DIR}/message.pb.cc"
  "${CMAKE_CURRENT_BINARY_DIR}/message.pb.h"
  "${CMAKE_CURRENT_BINARY_DIR}/message.grpc.pb.cc"
  "${CMAKE_CURRENT_BINARY_DIR}/message.grpc.pb.h")

target_link_libraries(
  sigrpc_grpc_proto
  absl::check
  ${_REFLECTION}
  ${_GRPC_GRPCPP}
  ${_PROTOBUF_LIBPROTOBUF})


foreach(_target
  sigrpc-runtime)
  add_executable(${_target} "${_target}.cc")
  set(LINKER_SCRIPT "${CMAKE_CURRENT_BINARY_DIR}/../elf64-x86-64.ld")
  set_target_properties(${_target} PROPERTIES LINK_FLAGS "-T${LINKER_SCRIPT}")
  target_link_libraries(${_target}
    sigrpc_grpc_proto
    absl::check
    absl::flags
    absl::flags_parse
    absl::log
    ${_REFLECTION}
    ${_GRPC_GRPCPP}
    ${_PROTOBUF_LIBPROTOBUF})
endforeach()

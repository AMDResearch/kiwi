// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

/*
 * invoke + ROCm-XIO SDMA ping-pong benchmark.
 *
 * Exactly two MPI ranks are used. Rank N maps to GPU N. Rank 0 reports
 * host-side amortized timing by default; optional --device-timing adds GPU
 * wall_clock64 per-iteration diagnostics.
 *
 * Modes:
 *   - invoke: GPU enqueues a invoke callback; the CPU callback immediately
 *               issues host-initiated ROCm-XIO SDMA put_signal() to the peer.
 *   - device:   device-initiated ROCm-XIO SDMA baseline, mirroring
 *               rocm-xio examples/sdma-ep-ping-pong.
 *   - host:     CPU-thread-initiated ROCm-XIO host SDMA ping-pong.
 *   - both:     run invoke, device, and host for easy comparison.
 */

#include "xio.h"
#include "endpoints/sdma-ep/sdma-ep.h"
#include "endpoints/sdma-ep/sdma-host-queue.h"
#include "endpoints/sdma-ep/sdma_device.hpp"

#ifdef HIP_CHECK
#undef HIP_CHECK
#endif

#include <kiwi/invoke/invoke.hpp>

#ifdef HIP_CHECK
#undef HIP_CHECK
#endif

#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>
#include <mpi.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <getopt.h>
#include <sched.h>
#include <limits>
#include <stdexcept>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace sdma_ep = xio::sdma_ep;

#define CHECK_HIP(call)                                                       \
  do {                                                                        \
    hipError_t _err = (call);                                                 \
    if (_err != hipSuccess) {                                                 \
      throw std::runtime_error(std::string(#call) + " failed: " +            \
                               hipGetErrorString(_err));                     \
    }                                                                         \
  } while (0)

static void clear_peer_access_status() {
  hipError_t err = hipGetLastError();
  if (err != hipSuccess && err != hipErrorPeerAccessAlreadyEnabled) {
    throw std::runtime_error(std::string("xio::enablePeerAccess failed: ") +
                             hipGetErrorString(err));
  }
}

struct BenchmarkArgs {
  int myRank = 0;
  int nRanks = 1;
  int myGpu = 0;
  int peerGpu = 1;
  std::string mode = "both";  // invoke, device, host, both
  std::vector<size_t> transferSizes{64, 256, 1024, 4096, 16384, 65536, 262144};
  int iterations = 1000;
  int warmup = 10;
  int minDurationMs = 1000;
  int queueCapacity = 4096;
  int pinCore = -1;
  int progressCore = -1;
  bool progressCoreExplicit = false;
  bool noPin = false;
  bool deviceTiming = false;
};

struct LatencyStats {
  std::string modeName;
  size_t transferSize = 0;
  std::string timingSource = "gpu_wall_clock64";
  bool hasCycles = true;
  unsigned long long minCycles = 0;
  unsigned long long maxCycles = 0;
  double avgCycles = 0.0;
  double minUs = 0.0;
  double maxUs = 0.0;
  double avgUs = 0.0;
};

struct PingPongBuffers {
  uint8_t* mySrcBuf = nullptr;
  uint8_t* myDstBuf = nullptr;
  uint64_t* mySignal = nullptr;
  uint8_t* peerSrcBuf = nullptr;
  uint8_t* peerDstBuf = nullptr;
  uint64_t* peerSignal = nullptr;
  unsigned long long* latencies = nullptr;
  size_t bufferSize = 0;
  int totalIters = 0;
};

struct HostSignalBuffers {
  MPI_Comm sharedComm = MPI_COMM_NULL;
  MPI_Win win = MPI_WIN_NULL;
  uint64_t* myCpuSignal = nullptr;
  uint64_t* peerCpuSignal = nullptr;
  uint64_t* mySdmaSignal = nullptr;
  uint64_t* peerSdmaSignal = nullptr;
  bool myRegistered = false;
  bool peerRegistered = false;
};

struct DeviceRunResult {
  double elapsedUs = 0.0;
  int iterations = 0;
  int payloadValidationFailed = 0;
  std::vector<unsigned long long> rank0Latencies;
};

struct InvokeRunResult {
  double elapsedUs = 0.0;
  int iterations = 0;
  int callbackErrors = 0;
  int payloadValidationFailed = 0;
  std::string proxyAffinity;
  std::vector<unsigned long long> rank0Latencies;
};

static double gpu_cycle_time_ns(int gpuId) {
  int clockRateKhz = 0;
  CHECK_HIP(hipDeviceGetAttribute(&clockRateKhz,
                                  hipDeviceAttributeWallClockRate, gpuId));
  return 1.0e6 / static_cast<double>(clockRateKhz);
}

static std::string affinity_mask_string() {
  cpu_set_t mask;
  CPU_ZERO(&mask);
  if (sched_getaffinity(0, sizeof(mask), &mask) != 0) {
    return std::string("unavailable: ") + std::strerror(errno);
  }

  std::ostringstream os;
  bool first = true;
  for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
    if (CPU_ISSET(cpu, &mask)) {
      if (!first) os << ",";
      os << cpu;
      first = false;
    }
  }
  if (first) return "empty";
  return os.str();
}

static std::string set_current_thread_affinity(int core) {
  cpu_set_t mask;
  CPU_ZERO(&mask);
  CPU_SET(core, &mask);
  if (sched_setaffinity(0, sizeof(mask), &mask) != 0) {
    return std::string("requested core ") + std::to_string(core) +
           " but sched_setaffinity failed: " + std::strerror(errno) +
           "; current CPUs: " + affinity_mask_string();
  }
  return std::string("pinned to core ") + std::to_string(core) +
         "; current CPUs: " + affinity_mask_string();
}

static std::string apply_cpu_affinity(const BenchmarkArgs& args) {
  if (args.noPin) {
    return "pinning disabled by --no-pin; current CPUs: " + affinity_mask_string();
  }
  if (args.pinCore < 0) {
    return "not pinned (use --pin-core/--progress-core); current CPUs: " +
           affinity_mask_string();
  }

  return std::string("main thread ") + set_current_thread_affinity(args.pinCore);
}

static void report_latency(const char* modeName, size_t transferSize,
                           const std::vector<unsigned long long>& latencies,
                           double cycleTimeNs,
                           std::vector<LatencyStats>& allStats) {
  unsigned long long minLat = std::numeric_limits<unsigned long long>::max();
  unsigned long long maxLat = 0;
  long double sum = 0.0;

  for (auto lat : latencies) {
    minLat = std::min(minLat, lat);
    maxLat = std::max(maxLat, lat);
    sum += static_cast<long double>(lat);
  }

  double avgCycles = static_cast<double>(sum / latencies.size());
  LatencyStats s;
  s.modeName = modeName;
  s.transferSize = transferSize;
  s.minCycles = minLat;
  s.maxCycles = maxLat;
  s.avgCycles = avgCycles;
  s.minUs = (static_cast<double>(minLat) * cycleTimeNs) / 1000.0;
  s.maxUs = (static_cast<double>(maxLat) * cycleTimeNs) / 1000.0;
  s.avgUs = (avgCycles * cycleTimeNs) / 1000.0;
  allStats.push_back(s);

  printf("%s [%zu bytes]: min %.3f us (%llu cycles), avg %.3f us "
         "(%.2f cycles), max %.3f us (%llu cycles)\n",
         modeName, transferSize, s.minUs, s.minCycles, s.avgUs, s.avgCycles,
         s.maxUs, s.maxCycles);
}

static void report_cpu_latency(const char* modeName, size_t transferSize,
                               const std::vector<double>& latenciesUs,
                               std::vector<LatencyStats>& allStats) {
  double minUs = std::numeric_limits<double>::max();
  double maxUs = 0.0;
  long double sum = 0.0;

  for (double lat : latenciesUs) {
    minUs = std::min(minUs, lat);
    maxUs = std::max(maxUs, lat);
    sum += static_cast<long double>(lat);
  }

  LatencyStats s;
  s.modeName = modeName;
  s.transferSize = transferSize;
  s.timingSource = "cpu_steady_clock";
  s.hasCycles = false;
  s.minUs = minUs;
  s.maxUs = maxUs;
  s.avgUs = static_cast<double>(sum / latenciesUs.size());
  allStats.push_back(s);

  printf("%s [%zu bytes]: min %.3f us, avg %.3f us, max %.3f us "
         "(timing: CPU steady_clock)\n",
         modeName, transferSize, s.minUs, s.avgUs, s.maxUs);
}

static uint64_t cpu_load_host_signal(uint64_t* signal) {
  return __atomic_load_n(signal, __ATOMIC_ACQUIRE);
}

static void cpu_store_host_signal(uint64_t* signal, uint64_t value) {
  __atomic_store_n(signal, value, __ATOMIC_RELEASE);
}

static void cpu_wait_for_host_signal(uint64_t* signal, uint64_t expected) {
  int idleSpins = 0;
  while (cpu_load_host_signal(signal) < expected) {
    if (++idleSpins % 256 == 0) {
      std::this_thread::yield();
    }
  }
}

static uint8_t source_pattern_for_rank(int rank) {
  return static_cast<uint8_t>(0xA0 + rank);
}

/* ROCm-XIO host SDMA revisions disagree on the formal argument names for
 * SdmaQueueHostHandle::put_signal(): some declare (src, dst, ...), while
 * others declare (dst, src, ...) and then submit CopyLinearPacket(src, dst).
 * CMake inspects the fetched source and defines
 * KIWI_QUEUE_ROCM_XIO_HOST_PUT_SIGNAL_SRC_DST to preserve the same semantic
 * direction at every host call site: local source buffer -> peer destination
 * buffer, then signal the peer. */
template <typename HostHandle, typename SignalT>
static void host_sdma_copy_local_src_to_peer_dst_and_signal(
    HostHandle& hostHandle, void* localSrc, void* peerDst, size_t size,
    SignalT* peerSignal, SignalT signalIncrement) {
#if KIWI_QUEUE_ROCM_XIO_HOST_PUT_SIGNAL_SRC_DST
  hostHandle.put_signal(localSrc, peerDst, size, peerSignal, signalIncrement);
#else
  hostHandle.put_signal(peerDst, localSrc, size, peerSignal, signalIncrement);
#endif
}

static bool should_validate_offset(size_t offset, size_t size) {
  if (size <= 4096) return true;
  if (offset == 0 || offset + 1 == size || offset == size / 2) return true;
  const size_t stride = std::max<size_t>(1, size / 64);
  return (offset % stride) == 0;
}

static int validate_payload_received(const BenchmarkArgs& args,
                                     const char* modeName,
                                     size_t transferSize,
                                     const PingPongBuffers& b) {
  std::vector<uint8_t> hostDst(transferSize);
  CHECK_HIP(hipMemcpy(hostDst.data(), b.myDstBuf, transferSize,
                      hipMemcpyDeviceToHost));

  const uint8_t expected = source_pattern_for_rank(1 - args.myRank);
  size_t checked = 0;
  size_t firstBad = transferSize;
  uint8_t firstBadValue = 0;
  for (size_t i = 0; i < transferSize; ++i) {
    if (!should_validate_offset(i, transferSize)) continue;
    ++checked;
    if (hostDst[i] != expected) {
      firstBad = i;
      firstBadValue = hostDst[i];
      break;
    }
  }

  int localOk = (firstBad == transferSize) ? 1 : 0;
  int globalOk = 0;
  MPI_Allreduce(&localOk, &globalOk, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

  if (!localOk) {
    fprintf(stderr,
            "[rank %d] payload validation FAILED for %s [%zu bytes]: "
            "myDstBuf[%zu]=0x%02x, expected peer rank %d source pattern 0x%02x "
            "(%zu sampled/checked bytes)\n",
            args.myRank, modeName, transferSize, firstBad, firstBadValue,
            1 - args.myRank, expected, checked);
  }
  if (args.myRank == 0) {
    if (globalOk) {
      printf("payload validation passed for %s [%zu bytes]: each rank's "
             "destination buffer contains the peer source pattern\n",
             modeName, transferSize);
    } else {
      fprintf(stderr, "payload validation FAILED for %s [%zu bytes]\n",
              modeName, transferSize);
    }
  }
  return globalOk ? 0 : 1;
}

static void setup_host_signal_buffers(const BenchmarkArgs& args,
                                      HostSignalBuffers& signals) {
  CHECK_HIP(hipSetDevice(args.myGpu));

  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, args.myRank,
                      MPI_INFO_NULL, &signals.sharedComm);
  int sharedSize = 0;
  int sharedRank = -1;
  MPI_Comm_size(signals.sharedComm, &sharedSize);
  MPI_Comm_rank(signals.sharedComm, &sharedRank);
  if (sharedSize != args.nRanks || sharedRank != args.myRank) {
    throw std::runtime_error(
        "host mode requires both MPI ranks in one shared-memory communicator");
  }

  MPI_Aint size = static_cast<MPI_Aint>(sizeof(uint64_t));
  int dispUnit = static_cast<int>(sizeof(uint64_t));
  void* base = nullptr;
  MPI_Win_allocate_shared(size, dispUnit, MPI_INFO_NULL, signals.sharedComm,
                          &base, &signals.win);
  signals.myCpuSignal = static_cast<uint64_t*>(base);
  if (signals.myCpuSignal == nullptr) {
    throw std::runtime_error("MPI_Win_allocate_shared returned null host signal");
  }
  cpu_store_host_signal(signals.myCpuSignal, 0);

  MPI_Aint peerSize = 0;
  int peerDispUnit = 0;
  void* peerBase = nullptr;
  MPI_Win_shared_query(signals.win, 1 - args.myRank, &peerSize,
                       &peerDispUnit, &peerBase);
  if (peerBase == nullptr || peerSize < static_cast<MPI_Aint>(sizeof(uint64_t))) {
    throw std::runtime_error("MPI_Win_shared_query returned invalid peer host signal");
  }
  signals.peerCpuSignal = static_cast<uint64_t*>(peerBase);

  MPI_Barrier(signals.sharedComm);

  hipError_t err = hipHostRegister(signals.myCpuSignal, sizeof(uint64_t),
                                   hipHostRegisterPortable |
                                       hipHostRegisterMapped);
  if (err != hipSuccess) {
    throw std::runtime_error(std::string("hipHostRegister(my host signal) failed: ") +
                             hipGetErrorString(err));
  }
  signals.myRegistered = true;

  err = hipHostRegister(signals.peerCpuSignal, sizeof(uint64_t),
                        hipHostRegisterPortable | hipHostRegisterMapped);
  if (err != hipSuccess) {
    throw std::runtime_error(std::string("hipHostRegister(peer host signal) failed: ") +
                             hipGetErrorString(err));
  }
  signals.peerRegistered = true;

  CHECK_HIP(hipHostGetDevicePointer(reinterpret_cast<void**>(&signals.mySdmaSignal),
                                    signals.myCpuSignal, 0));
  CHECK_HIP(hipHostGetDevicePointer(reinterpret_cast<void**>(&signals.peerSdmaSignal),
                                    signals.peerCpuSignal, 0));
  MPI_Barrier(signals.sharedComm);
}

static void cleanup_host_signal_buffers(HostSignalBuffers& signals) {
  if (signals.peerRegistered) {
    CHECK_HIP(hipHostUnregister(signals.peerCpuSignal));
    signals.peerRegistered = false;
  }
  if (signals.myRegistered) {
    CHECK_HIP(hipHostUnregister(signals.myCpuSignal));
    signals.myRegistered = false;
  }
  if (signals.win != MPI_WIN_NULL) {
    MPI_Win_free(&signals.win);
    signals.win = MPI_WIN_NULL;
  }
  if (signals.sharedComm != MPI_COMM_NULL) {
    MPI_Comm_free(&signals.sharedComm);
    signals.sharedComm = MPI_COMM_NULL;
  }
}

static void setup_buffers(const BenchmarkArgs& args, size_t transferSize,
                          int totalIters, bool allocateLatencies,
                          PingPongBuffers& b, bool openPeerOnPeerDevice) {
  CHECK_HIP(hipSetDevice(args.myGpu));
  b.bufferSize = transferSize;
  b.totalIters = totalIters;

  CHECK_HIP(hipExtMallocWithFlags(reinterpret_cast<void**>(&b.mySrcBuf),
                                  transferSize, hipDeviceMallocUncached));
  CHECK_HIP(hipMemset(b.mySrcBuf, source_pattern_for_rank(args.myRank), transferSize));
  CHECK_HIP(hipExtMallocWithFlags(reinterpret_cast<void**>(&b.myDstBuf),
                                  transferSize, hipDeviceMallocUncached));
  CHECK_HIP(hipMemset(b.myDstBuf, 0, transferSize));
  CHECK_HIP(hipExtMallocWithFlags(reinterpret_cast<void**>(&b.mySignal),
                                  sizeof(uint64_t), hipDeviceMallocUncached));
  CHECK_HIP(hipMemset(b.mySignal, 0, sizeof(uint64_t)));

  hipIpcMemHandle_t mySrcHandle, myDstHandle, mySignalHandle;
  CHECK_HIP(hipIpcGetMemHandle(&mySrcHandle, b.mySrcBuf));
  CHECK_HIP(hipIpcGetMemHandle(&myDstHandle, b.myDstBuf));
  CHECK_HIP(hipIpcGetMemHandle(&mySignalHandle, b.mySignal));

  hipIpcMemHandle_t peerSrcHandle, peerDstHandle, peerSignalHandle;
  MPI_Sendrecv(&mySrcHandle, sizeof(mySrcHandle), MPI_BYTE, 1 - args.myRank, 10,
               &peerSrcHandle, sizeof(peerSrcHandle), MPI_BYTE,
               1 - args.myRank, 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  MPI_Sendrecv(&myDstHandle, sizeof(myDstHandle), MPI_BYTE, 1 - args.myRank, 11,
               &peerDstHandle, sizeof(peerDstHandle), MPI_BYTE,
               1 - args.myRank, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  MPI_Sendrecv(&mySignalHandle, sizeof(mySignalHandle), MPI_BYTE,
               1 - args.myRank, 12, &peerSignalHandle,
               sizeof(peerSignalHandle), MPI_BYTE, 1 - args.myRank, 12,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  int prevDevice = -1;
  if (openPeerOnPeerDevice) {
    CHECK_HIP(hipGetDevice(&prevDevice));
    CHECK_HIP(hipSetDevice(args.peerGpu));
  }
  CHECK_HIP(hipIpcOpenMemHandle(reinterpret_cast<void**>(&b.peerSrcBuf),
                                peerSrcHandle,
                                hipIpcMemLazyEnablePeerAccess));
  CHECK_HIP(hipIpcOpenMemHandle(reinterpret_cast<void**>(&b.peerDstBuf),
                                peerDstHandle,
                                hipIpcMemLazyEnablePeerAccess));
  CHECK_HIP(hipIpcOpenMemHandle(reinterpret_cast<void**>(&b.peerSignal),
                                peerSignalHandle,
                                hipIpcMemLazyEnablePeerAccess));
  if (openPeerOnPeerDevice && prevDevice != args.peerGpu) {
    CHECK_HIP(hipSetDevice(prevDevice));
  }

  if (allocateLatencies && args.myRank == 0) {
    CHECK_HIP(hipMalloc(&b.latencies,
                        b.totalIters * sizeof(unsigned long long)));
  }
}

static void cleanup_buffers(PingPongBuffers& b) {
  if (b.peerSrcBuf) CHECK_HIP(hipIpcCloseMemHandle(b.peerSrcBuf));
  if (b.peerDstBuf) CHECK_HIP(hipIpcCloseMemHandle(b.peerDstBuf));
  if (b.peerSignal) CHECK_HIP(hipIpcCloseMemHandle(b.peerSignal));
  if (b.mySrcBuf) CHECK_HIP(hipFree(b.mySrcBuf));
  if (b.myDstBuf) CHECK_HIP(hipFree(b.myDstBuf));
  if (b.mySignal) CHECK_HIP(hipFree(b.mySignal));
  if (b.latencies) CHECK_HIP(hipFree(b.latencies));
}

__global__ void device_ping_kernel(sdma_ep::SdmaQueueHandle* myQueue,
                                   void* myBuf, void* peerBuf, size_t size,
                                   uint64_t* mySignal,
                                   uint64_t* peerSignal,
                                   unsigned long long* latencies,
                                   int iterations) {
  sdma_ep::SdmaQueueHandle q = *myQueue;
  for (int i = 0; i < iterations; ++i) {
    unsigned long long t0 = wall_clock64();
    sdma_ep::putSignal(q, peerBuf, myBuf, size, peerSignal);
    sdma_ep::waitSignal(mySignal, static_cast<uint64_t>(i + 1));
    if (latencies != nullptr) latencies[i] = wall_clock64() - t0;
  }
}

__global__ void device_pong_kernel(sdma_ep::SdmaQueueHandle* myQueue,
                                   void* myBuf, void* peerBuf, size_t size,
                                   uint64_t* mySignal,
                                   uint64_t* peerSignal,
                                   int iterations) {
  sdma_ep::SdmaQueueHandle q = *myQueue;
  for (int i = 0; i < iterations; ++i) {
    sdma_ep::waitSignal(mySignal, static_cast<uint64_t>(i + 1));
    sdma_ep::putSignal(q, peerBuf, myBuf, size, peerSignal);
  }
}

__global__ void invoke_ping_kernel(kiwi::invoke::DeviceContext<>* ctx,
                                     uint8_t callbackId,
                                     uint64_t* mySignal,
                                     unsigned long long* latencies,
                                     int iterations) {
  auto handle = ctx->get_device_handle(0);
  for (int i = 0; i < iterations; ++i) {
    unsigned long long t0 = wall_clock64();
    while (!handle.invoke(callbackId, static_cast<uint32_t>(i + 1))) {
      /* retry while callback queue is full */
    }
    /* The timed region intentionally ends after the peer callback's SDMA
     * response updates mySignal, not after callback enqueue/proxy overlap. */
    sdma_ep::waitSignal(mySignal, static_cast<uint64_t>(i + 1));
    unsigned long long dt = wall_clock64() - t0;
    if (latencies != nullptr && threadIdx.x == 0) latencies[i] = dt;
  }
}

__global__ void invoke_pong_kernel(kiwi::invoke::DeviceContext<>* ctx,
                                     uint8_t callbackId,
                                     uint64_t* mySignal,
                                     int iterations) {
  auto handle = ctx->get_device_handle(0);
  for (int i = 0; i < iterations; ++i) {
    sdma_ep::waitSignal(mySignal, static_cast<uint64_t>(i + 1));
    while (!handle.invoke(callbackId, static_cast<uint32_t>(i + 1))) {
      /* retry while callback queue is full */
    }
  }
}

static void report_amortized_host_timing(const char* modeName,
                                        const char* detail,
                                        size_t transferSize,
                                        int iterations,
                                        double elapsedUs,
                                        int minDurationMs,
                                        std::vector<LatencyStats>& allStats) {
  LatencyStats s;
  s.modeName = modeName;
  s.transferSize = transferSize;
  s.timingSource = "cpu_steady_clock";
  s.hasCycles = false;
  s.minUs = elapsedUs / static_cast<double>(iterations);
  s.maxUs = s.minUs;
  s.avgUs = s.minUs;
  allStats.push_back(s);

  printf("%s [%zu bytes]: avg %.3f us over %d iterations "
         "(total %.3f ms, timing: CPU steady_clock %s; min-duration-ms=%d)\n",
         modeName, transferSize, s.avgUs, iterations, elapsedUs / 1000.0,
         detail, minDurationMs);
}

static DeviceRunResult run_device_batch(const BenchmarkArgs& args,
                                          sdma_ep::SdmaQueueInfo& queueInfo,
                                          size_t size,
                                          int iterations,
                                          bool collectDeviceTiming,
                                          bool validatePayload) {
  DeviceRunResult result;
  result.iterations = iterations;

  PingPongBuffers b;
  setup_buffers(args, size, iterations, collectDeviceTiming, b, false);
  MPI_Barrier(MPI_COMM_WORLD);

  auto t0 = std::chrono::steady_clock::now();
  if (args.myRank == 0) {
    hipLaunchKernelGGL(device_ping_kernel, dim3(1), dim3(1), 0, nullptr,
                       static_cast<sdma_ep::SdmaQueueHandle*>(queueInfo.deviceHandle),
                       b.mySrcBuf, b.peerDstBuf, size, b.mySignal,
                       b.peerSignal, b.latencies, iterations);
  } else {
    hipLaunchKernelGGL(device_pong_kernel, dim3(1), dim3(1), 0, nullptr,
                       static_cast<sdma_ep::SdmaQueueHandle*>(queueInfo.deviceHandle),
                       b.mySrcBuf, b.peerDstBuf, size, b.mySignal,
                       b.peerSignal, iterations);
  }
  CHECK_HIP(hipGetLastError());
  CHECK_HIP(hipDeviceSynchronize());
  auto t1 = std::chrono::steady_clock::now();

  std::chrono::duration<double, std::micro> dt = t1 - t0;
  double localElapsedUs = dt.count();
  double maxElapsedUs = 0.0;
  MPI_Allreduce(&localElapsedUs, &maxElapsedUs, 1, MPI_DOUBLE, MPI_MAX,
                MPI_COMM_WORLD);
  result.elapsedUs = maxElapsedUs;

  if (collectDeviceTiming && args.myRank == 0) {
    result.rank0Latencies.resize(iterations);
    CHECK_HIP(hipMemcpy(result.rank0Latencies.data(), b.latencies,
                        iterations * sizeof(unsigned long long),
                        hipMemcpyDeviceToHost));
  }

  MPI_Barrier(MPI_COMM_WORLD);
  if (validatePayload) {
    result.payloadValidationFailed = validate_payload_received(
        args, "device-initiated-host-amortized", size, b);
  }
  cleanup_buffers(b);
  return result;
}

static int run_device_baseline(const BenchmarkArgs& args,
                               std::vector<LatencyStats>& allStats) {
  if (args.myRank == 0) {
    printf("\n=== Device-initiated ROCm-XIO SDMA baseline ===\n");
    printf("Primary timing source: CPU steady_clock around kernel launch/synchronize, "
           "amortized over iterations.\n");
    if (args.deviceTiming) {
      printf("Device timing diagnostics: enabled (rank-0 GPU wall_clock64 per iteration).\n");
    }
  }

  xio::enablePeerAccess(args.myGpu, args.peerGpu);
  clear_peer_access_status();
  sdma_ep::SdmaQueueInfo queueInfo = {};
  int rc = sdma_ep::createQueue(args.myGpu, args.peerGpu, &queueInfo);
  if (rc != 0) {
    fprintf(stderr, "[rank %d] createQueue(%d->%d) failed (%d)\n",
            args.myRank, args.myGpu, args.peerGpu, rc);
    return 1;
  }

  for (size_t size : args.transferSizes) {
    if (args.warmup > 0) {
      (void)run_device_batch(args, queueInfo, size, args.warmup, false, false);
      if (args.myRank == 0) {
        printf("device warmup [%zu bytes]: %d iterations\n", size, args.warmup);
      }
    }

    int timedIterations = args.iterations;
    constexpr int kMaxCalibrationAttempts = 8;
    DeviceRunResult measured;
    for (int attempt = 0; attempt < kMaxCalibrationAttempts; ++attempt) {
      measured = run_device_batch(args, queueInfo, size, timedIterations,
                                  args.deviceTiming, true);
      if (measured.payloadValidationFailed != 0) {
        sdma_ep::destroyQueue(&queueInfo);
        return 1;
      }

      const double targetUs = static_cast<double>(args.minDurationMs) * 1000.0;
      if (args.minDurationMs <= 0 || measured.elapsedUs >= targetUs) {
        break;
      }

      int nextIterations = static_cast<int>(
          (static_cast<long double>(timedIterations) * targetUs * 1.25L) /
          std::max(1.0, measured.elapsedUs));
      nextIterations = std::max(timedIterations * 2, nextIterations);
      nextIterations = std::max(timedIterations + 1, nextIterations);
      if (attempt + 1 == kMaxCalibrationAttempts) {
        if (args.myRank == 0) {
          printf("device calibration [%zu bytes]: WARNING %d iterations took %.3f ms, "
                 "below --min-duration-ms=%d after %d attempts; reporting this run\n",
                 size, timedIterations, measured.elapsedUs / 1000.0,
                 args.minDurationMs, kMaxCalibrationAttempts);
        }
        break;
      }
      if (args.myRank == 0) {
        printf("device calibration [%zu bytes]: %d iterations took %.3f ms; "
               "increasing timed iterations to %d for --min-duration-ms=%d\n",
               size, timedIterations, measured.elapsedUs / 1000.0,
               nextIterations, args.minDurationMs);
      }
      timedIterations = nextIterations;
    }

    if (args.myRank == 0) {
      report_amortized_host_timing(
          "device-initiated-host-amortized",
          "around device kernel launch and synchronize",
          size, measured.iterations, measured.elapsedUs, args.minDurationMs,
          allStats);
      if (args.deviceTiming) {
        report_latency("device-initiated-gpu-diagnostic", size,
                       measured.rank0Latencies, gpu_cycle_time_ns(args.myGpu),
                       allStats);
      }
    }
  }

  sdma_ep::destroyQueue(&queueInfo);
  return 0;
}

static int run_host_sdma_baseline(const BenchmarkArgs& args,
                                  std::vector<LatencyStats>& allStats) {
  if (args.myRank == 0) {
    printf("\n=== Pure host-side ROCm-XIO SDMA baseline ===\n");
    printf("Timing source: CPU steady_clock around rank-0 round trip; "
           "CPU waits directly poll host-pinned CPU-visible signal memory.\n");
  }

  xio::enablePeerAccess(args.myGpu, args.peerGpu);
  clear_peer_access_status();
  sdma_ep::SdmaQueueInfo queueInfo = {};
  int rc = sdma_ep::createHostQueue(args.myGpu, args.peerGpu, &queueInfo);
  if (rc != 0) {
    fprintf(stderr, "[rank %d] createHostQueue(%d->%d) failed (%d)\n",
            args.myRank, args.myGpu, args.peerGpu, rc);
    return 1;
  }
  auto hostHandle = sdma_ep::getHostHandle(queueInfo.srcDeviceId,
                                           queueInfo.dstDeviceId, 0);

  for (size_t size : args.transferSizes) {
    PingPongBuffers b;
    setup_buffers(args, size, args.warmup + args.iterations, false, b, true);
    HostSignalBuffers hostSignals;
    setup_host_signal_buffers(args, hostSignals);
    MPI_Barrier(MPI_COMM_WORLD);

    std::vector<double> latenciesUs;
    if (args.myRank == 0) latenciesUs.reserve(args.iterations);

    for (int i = 0; i < b.totalIters; ++i) {
      const uint64_t expected = static_cast<uint64_t>(i + 1);
      if (args.myRank == 0) {
        auto t0 = std::chrono::steady_clock::now();
        host_sdma_copy_local_src_to_peer_dst_and_signal(
            hostHandle, b.mySrcBuf, b.peerDstBuf, size,
            hostSignals.peerSdmaSignal, uint64_t{1});
        cpu_wait_for_host_signal(hostSignals.myCpuSignal, expected);
        if (i >= args.warmup) {
          auto t1 = std::chrono::steady_clock::now();
          std::chrono::duration<double, std::micro> dt = t1 - t0;
          latenciesUs.push_back(dt.count());
        }
      } else {
        cpu_wait_for_host_signal(hostSignals.myCpuSignal, expected);
        host_sdma_copy_local_src_to_peer_dst_and_signal(
            hostHandle, b.mySrcBuf, b.peerDstBuf, size,
            hostSignals.peerSdmaSignal, uint64_t{1});
      }
    }

    hostHandle.quiet();
    MPI_Barrier(MPI_COMM_WORLD);

    if (args.myRank == 0) {
      report_cpu_latency("host-sdma-host-pinned-signal-cpu-timed", size,
                         latenciesUs, allStats);
    }
    if (validate_payload_received(args,
                                  "host-sdma-host-pinned-signal-cpu-timed",
                                  size, b) != 0) {
      cleanup_host_signal_buffers(hostSignals);
      cleanup_buffers(b);
      sdma_ep::destroyQueue(&queueInfo);
      return 1;
    }
    cleanup_host_signal_buffers(hostSignals);
    cleanup_buffers(b);
  }

  sdma_ep::destroyQueue(&queueInfo);
  return 0;
}

template <typename HostHandle>
static InvokeRunResult run_invoke_batch(const BenchmarkArgs& args,
                                            HostHandle& hostHandle,
                                            size_t size,
                                            int iterations,
                                            bool measuredRun,
                                            bool collectDeviceTiming) {
  InvokeRunResult result;
  result.iterations = iterations;

  PingPongBuffers b;
  setup_buffers(args, size, iterations, collectDeviceTiming, b, true);

  kiwi::invoke::Context<> ctx(1, static_cast<size_t>(args.queueCapacity));
  std::atomic<bool> stopProxy{false};
  std::atomic<int> callbackErrors{0};
  std::atomic<int> proxyReady{0};
  uint32_t expectedSeq = 1;

  std::function<void(uint32_t)> proxyCallback = [&](uint32_t seq) {
    if (seq != expectedSeq) {
      callbackErrors.fetch_add(1, std::memory_order_relaxed);
      expectedSeq = seq;
    }
    ++expectedSeq;
    host_sdma_copy_local_src_to_peer_dst_and_signal(
        hostHandle, b.mySrcBuf, b.peerDstBuf, size, b.peerSignal, uint64_t{1});
  };
  uint8_t cb = ctx.register_callback<uint32_t>(std::move(proxyCallback));

  std::exception_ptr proxyException;
  std::thread proxyThread([&] {
    try {
      if (!args.noPin && args.progressCore >= 0) {
        result.proxyAffinity = std::string("proxy thread ") +
                               set_current_thread_affinity(args.progressCore);
      } else {
        result.proxyAffinity = std::string("proxy thread not pinned; current CPUs: ") +
                               affinity_mask_string();
      }
      proxyReady.store(1, std::memory_order_release);
      int idleSpins = 0;
      while (!stopProxy.load(std::memory_order_acquire)) {
        size_t progressed = ctx.progress(0, 1);
        if (progressed == 0) {
          if (++idleSpins % 256 == 0) std::this_thread::yield();
        } else {
          idleSpins = 0;
        }
      }
      for (;;) {
        size_t progressed = ctx.progress(0, -1);
        if (progressed == 0) break;
      }
    } catch (...) {
      proxyException = std::current_exception();
      proxyReady.store(1, std::memory_order_release);
    }
  });

  while (proxyReady.load(std::memory_order_acquire) == 0) {
    std::this_thread::yield();
  }

  MPI_Barrier(MPI_COMM_WORLD);

  hipStream_t stream = nullptr;
  CHECK_HIP(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));

  auto t0 = std::chrono::steady_clock::now();
  try {
    if (args.myRank == 0) {
      hipLaunchKernelGGL(invoke_ping_kernel, dim3(1), dim3(64), 0, stream,
                         ctx.device_context(), cb, b.mySignal, b.latencies,
                         iterations);
    } else {
      hipLaunchKernelGGL(invoke_pong_kernel, dim3(1), dim3(64), 0, stream,
                         ctx.device_context(), cb, b.mySignal, iterations);
    }
    CHECK_HIP(hipGetLastError());
    CHECK_HIP(hipStreamSynchronize(stream));
    CHECK_HIP(hipStreamDestroy(stream));
    stream = nullptr;
    stopProxy.store(true, std::memory_order_release);
    proxyThread.join();
    if (proxyException) std::rethrow_exception(proxyException);
  } catch (...) {
    stopProxy.store(true, std::memory_order_release);
    if (proxyThread.joinable()) proxyThread.join();
    if (stream != nullptr) (void)hipStreamDestroy(stream);
    cleanup_buffers(b);
    throw;
  }
  hostHandle.quiet();
  auto t1 = std::chrono::steady_clock::now();

  if (measuredRun) {
    std::chrono::duration<double, std::micro> dt = t1 - t0;
    result.elapsedUs = dt.count();
    if (collectDeviceTiming && args.myRank == 0) {
      result.rank0Latencies.resize(iterations);
      CHECK_HIP(hipMemcpy(result.rank0Latencies.data(), b.latencies,
                          iterations * sizeof(unsigned long long),
                          hipMemcpyDeviceToHost));
    }
    result.payloadValidationFailed = validate_payload_received(
        args, "invoke-host-sdma-host-amortized", size, b);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  result.callbackErrors = callbackErrors.load(std::memory_order_relaxed);
  cleanup_buffers(b);
  return result;
}

static int run_invoke_reverse_offload(const BenchmarkArgs& args,
                                        std::vector<LatencyStats>& allStats) {
  if (args.myRank == 0) {
    printf("\n=== invoke reverse offload + host-initiated ROCm-XIO SDMA ===\n");
    printf("Primary timing source: CPU steady_clock amortized over the whole "
           "kernel/proxy/host-SDMA run.\n");
    if (args.deviceTiming) {
      printf("Device timing diagnostics: enabled (rank-0 GPU wall_clock64 per iteration "
             "from invoke invoke until peer response signal is observed).\n");
    }
  }

  xio::enablePeerAccess(args.myGpu, args.peerGpu);
  clear_peer_access_status();
  sdma_ep::SdmaQueueInfo queueInfo = {};
  int rc = sdma_ep::createHostQueue(args.myGpu, args.peerGpu, &queueInfo);
  if (rc != 0) {
    fprintf(stderr, "[rank %d] createHostQueue(%d->%d) failed (%d)\n",
            args.myRank, args.myGpu, args.peerGpu, rc);
    return 1;
  }
  auto hostHandle = sdma_ep::getHostHandle(queueInfo.srcDeviceId,
                                           queueInfo.dstDeviceId, 0);

  for (size_t size : args.transferSizes) {
    if (args.warmup > 0) {
      InvokeRunResult warm = run_invoke_batch(args, hostHandle, size,
                                                  args.warmup, false, false);
      int warmErrors = warm.callbackErrors;
      int globalWarmErrors = 0;
      MPI_Allreduce(&warmErrors, &globalWarmErrors, 1, MPI_INT, MPI_MAX,
                    MPI_COMM_WORLD);
      if (globalWarmErrors != 0) {
        fprintf(stderr, "[rank %d] invoke warmup callback sequence errors: %d\n",
                args.myRank, warm.callbackErrors);
        sdma_ep::destroyQueue(&queueInfo);
        return 1;
      }
      if (args.myRank == 0) {
        printf("invoke warmup [%zu bytes]: %d iterations; %s\n",
               size, args.warmup, warm.proxyAffinity.c_str());
      }
    }

    int timedIterations = args.iterations;
    constexpr int kMaxCalibrationAttempts = 8;
    InvokeRunResult measured;
    for (int attempt = 0; attempt < kMaxCalibrationAttempts; ++attempt) {
      measured = run_invoke_batch(args, hostHandle, size,
                                    timedIterations, true,
                                    args.deviceTiming);
      int localErrors = measured.callbackErrors;
      int globalErrors = 0;
      MPI_Allreduce(&localErrors, &globalErrors, 1, MPI_INT, MPI_MAX,
                    MPI_COMM_WORLD);
      if (globalErrors != 0) {
        fprintf(stderr, "[rank %d] invoke callback sequence errors: %d\n",
                args.myRank, measured.callbackErrors);
        sdma_ep::destroyQueue(&queueInfo);
        return 1;
      }
      if (measured.payloadValidationFailed != 0) {
        sdma_ep::destroyQueue(&queueInfo);
        return 1;
      }
      double localElapsedUs = measured.elapsedUs;
      double maxElapsedUs = 0.0;
      MPI_Allreduce(&localElapsedUs, &maxElapsedUs, 1, MPI_DOUBLE, MPI_MAX,
                    MPI_COMM_WORLD);
      measured.elapsedUs = maxElapsedUs;

      const double targetUs = static_cast<double>(args.minDurationMs) * 1000.0;
      if (args.minDurationMs <= 0 || measured.elapsedUs >= targetUs) {
        break;
      }

      int nextIterations = static_cast<int>(
          (static_cast<long double>(timedIterations) * targetUs * 1.25L) /
          std::max(1.0, measured.elapsedUs));
      nextIterations = std::max(timedIterations * 2, nextIterations);
      nextIterations = std::max(timedIterations + 1, nextIterations);
      if (attempt + 1 == kMaxCalibrationAttempts) {
        if (args.myRank == 0) {
          printf("invoke calibration [%zu bytes]: WARNING %d iterations took %.3f ms, "
                 "below --min-duration-ms=%d after %d attempts; reporting this run\n",
                 size, timedIterations, measured.elapsedUs / 1000.0,
                 args.minDurationMs, kMaxCalibrationAttempts);
        }
        break;
      }
      if (args.myRank == 0) {
        printf("invoke calibration [%zu bytes]: %d iterations took %.3f ms; "
               "increasing timed iterations to %d for --min-duration-ms=%d\n",
               size, timedIterations, measured.elapsedUs / 1000.0,
               nextIterations, args.minDurationMs);
      }
      timedIterations = nextIterations;
    }

    if (args.myRank == 0) {
      printf("[rank 0] %s\n", measured.proxyAffinity.c_str());
      report_amortized_host_timing(
          "invoke-host-sdma-host-amortized",
          "around kernel launch, completion, proxy stop/join/drain, and host SDMA quiet",
          size, measured.iterations, measured.elapsedUs, args.minDurationMs,
          allStats);
      if (args.deviceTiming) {
        report_latency("invoke-host-sdma-gpu-diagnostic", size,
                       measured.rank0Latencies, gpu_cycle_time_ns(args.myGpu),
                       allStats);
      }
    }
  }

  sdma_ep::destroyQueue(&queueInfo);
  return 0;
}

static std::vector<size_t> parse_sizes(const char* text) {
  std::vector<size_t> sizes;
  const char* p = text;
  while (*p) {
    char* end = nullptr;
    unsigned long long v = strtoull(p, &end, 10);
    if (end == p || v == 0) {
      throw std::runtime_error(std::string("invalid --sizes value: ") + text);
    }
    sizes.push_back(static_cast<size_t>(v));
    if (*end == '\0') break;
    if (*end != ',') {
      throw std::runtime_error(std::string("invalid --sizes value: ") + text);
    }
    p = end + 1;
  }
  return sizes;
}

static void print_usage(const char* prog) {
  if (prog == nullptr) prog = "invoke_sdma_ping_pong";
  printf("Usage: mpirun -n 2 %s [options]\n", prog);
  printf("Options:\n");
  printf("  --mode <invoke|device|host|both|all>  Benchmark mode (default: both; both/all run all three)\n");
  printf("  --iterations <N>              Initial measured iterations (default: 1000)\n");
  printf("  --warmup <N>                  Untimed warmup iterations before reporting (default: 10)\n");
  printf("  --min-duration-ms <N>         Increase kiwi/invoke/device iterations until host amortized timing reaches N ms (default: 1000; 0 disables)\n");
  printf("  --device-timing               Also print GPU wall_clock64 per-iteration diagnostic rows (default off)\n");
  printf("  --gpu-timing                  Alias for --device-timing\n");
  printf("  --sizes <B[,B...]>            Transfer sizes in bytes\n");
  printf("  --queue-capacity <N>          invoke queue capacity (default: 4096)\n");
  printf("  --pin-core <N>                Pin the main thread to CPU core N; proxy defaults to same core\n");
  printf("  --progress-core <N>           Pin the invoke proxy/progress thread to CPU core N\n");
  printf("  --no-pin                      Do not set CPU affinity (default)\n");
  printf("  --help                        Show this message\n");
}

static BenchmarkArgs parse_args(int argc, char** argv) {
  BenchmarkArgs args;
  enum { OPT_PIN_CORE = 1000, OPT_PROGRESS_CORE, OPT_NO_PIN, OPT_MIN_DURATION_MS, OPT_DEVICE_TIMING };
  static option opts[] = {{"mode", required_argument, nullptr, 'm'},
                          {"iterations", required_argument, nullptr, 'i'},
                          {"warmup", required_argument, nullptr, 'w'},
                          {"sizes", required_argument, nullptr, 's'},
                          {"queue-capacity", required_argument, nullptr, 'q'},
                          {"min-duration-ms", required_argument, nullptr, OPT_MIN_DURATION_MS},
                          {"device-timing", no_argument, nullptr, OPT_DEVICE_TIMING},
                          {"gpu-timing", no_argument, nullptr, OPT_DEVICE_TIMING},
                          {"pin-core", required_argument, nullptr, OPT_PIN_CORE},
                          {"progress-core", required_argument, nullptr, OPT_PROGRESS_CORE},
                          {"no-pin", no_argument, nullptr, OPT_NO_PIN},
                          {"help", no_argument, nullptr, 'h'},
                          {nullptr, 0, nullptr, 0}};

  int c;
  while ((c = getopt_long(argc, argv, "m:i:w:s:q:h", opts, nullptr)) != -1) {
    switch (c) {
    case 'm':
      args.mode = optarg;
      if (args.mode == "all") args.mode = "both";
      if (args.mode != "invoke" && args.mode != "device" &&
          args.mode != "host" && args.mode != "both") {
        throw std::runtime_error("--mode must be invoke, device, host, both, or all");
      }
      break;
    case 'i':
      args.iterations = std::max(1, atoi(optarg));
      break;
    case 'w':
      args.warmup = std::max(0, atoi(optarg));
      break;
    case 's':
      args.transferSizes = parse_sizes(optarg);
      break;
    case 'q':
      args.queueCapacity = std::max(64, atoi(optarg));
      break;
    case OPT_MIN_DURATION_MS:
      args.minDurationMs = std::max(0, atoi(optarg));
      break;
    case OPT_DEVICE_TIMING:
      args.deviceTiming = true;
      break;
    case OPT_PIN_CORE:
      args.pinCore = std::max(0, atoi(optarg));
      if (!args.progressCoreExplicit) args.progressCore = args.pinCore;
      args.noPin = false;
      break;
    case OPT_PROGRESS_CORE:
      args.progressCore = std::max(0, atoi(optarg));
      args.progressCoreExplicit = true;
      if (args.pinCore < 0) args.pinCore = args.progressCore;
      args.noPin = false;
      break;
    case OPT_NO_PIN:
      args.noPin = true;
      args.pinCore = -1;
      args.progressCore = -1;
      break;
    case 'h':
      print_usage(argv[0]);
      exit(0);
    default:
      print_usage(argv[0]);
      exit(1);
    }
  }
  return args;
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  try {
    int nRanks = 0, myRank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

    BenchmarkArgs args = parse_args(argc, argv);
    args.nRanks = nRanks;
    args.myRank = myRank;
    args.myGpu = myRank;
    args.peerGpu = 1 - myRank;

    std::string affinityState = apply_cpu_affinity(args);

    if (nRanks != 2) {
      if (myRank == 0) {
        fprintf(stderr, "ERROR: this benchmark requires exactly 2 MPI ranks\n");
        print_usage(argv[0]);
      }
      MPI_Finalize();
      return 1;
    }

    int deviceCount = 0;
    CHECK_HIP(hipGetDeviceCount(&deviceCount));
    if (deviceCount < 2) {
      if (myRank == 0) printf("SKIP: need >=2 GPUs, found %d\n", deviceCount);
      MPI_Finalize();
      return 0;
    }

    CHECK_HIP(hipSetDevice(args.myGpu));
    int canAccess = 0;
    CHECK_HIP(hipDeviceCanAccessPeer(&canAccess, args.myGpu, args.peerGpu));
    if (!canAccess) {
      if (myRank == 0) {
        printf("SKIP: GPU %d cannot access GPU %d\n", args.myGpu,
               args.peerGpu);
      }
      MPI_Finalize();
      return 0;
    }

    hipDeviceProp_t prop;
    CHECK_HIP(hipGetDeviceProperties(&prop, args.myGpu));
    printf("[rank %d] GPU %d: %s\n", myRank, args.myGpu, prop.name);
    printf("[rank %d] CPU affinity: %s\n", myRank, affinityState.c_str());

    if (myRank == 0) {
      printf("Mode: %s, iterations: %d, warmup: %d, min-duration-ms: %d, queue capacity: %d, device timing: %s\n",
             args.mode.c_str(), args.iterations, args.warmup,
             args.minDurationMs, args.queueCapacity, args.deviceTiming ? "on" : "off");
      printf("Transfer sizes:");
      for (size_t s : args.transferSizes) printf(" %zu", s);
      printf(" bytes\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    int rc = sdma_ep::initEndpoint();
    if (rc != 0) {
      fprintf(stderr, "[rank %d] initEndpoint failed (%d)\n", myRank, rc);
      MPI_Abort(MPI_COMM_WORLD, 1);
    }

    std::vector<LatencyStats> stats;
    int ret = 0;
    if (args.mode == "invoke" || args.mode == "both") {
      ret = run_invoke_reverse_offload(args, stats);
      CHECK_HIP(hipSetDevice(args.myGpu));
      CHECK_HIP(hipDeviceSynchronize());
      MPI_Barrier(MPI_COMM_WORLD);
    }
    if (ret == 0 && (args.mode == "device" || args.mode == "both")) {
      ret = run_device_baseline(args, stats);
      CHECK_HIP(hipSetDevice(args.myGpu));
      CHECK_HIP(hipDeviceSynchronize());
      MPI_Barrier(MPI_COMM_WORLD);
    }
    if (ret == 0 && (args.mode == "host" || args.mode == "both")) {
      ret = run_host_sdma_baseline(args, stats);
    }

    sdma_ep::shutdownEndpoint();

    if (myRank == 0 && args.mode == "both" && stats.size() >= 3) {
      printf("\nSummary (lower is better):\n");
      printf("  %-36s %12s %14s  %s\n", "mode", "bytes", "avg_us", "timing");
      for (const auto& s : stats) {
        printf("  %-36s %12zu %14.3f  %s\n", s.modeName.c_str(),
               s.transferSize, s.avgUs, s.timingSource.c_str());
      }
      printf("Note: host-amortized CPU steady_clock rows are the default primary "
             "results. GPU wall_clock64 rows are optional diagnostics enabled by "
             "--device-timing and should be close on long enough runs.\n");
    }

    MPI_Finalize();
    return ret;
  } catch (const std::exception& ex) {
    int rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    fprintf(stderr, "[rank %d] ERROR: %s\n", rank, ex.what());
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
  }
}

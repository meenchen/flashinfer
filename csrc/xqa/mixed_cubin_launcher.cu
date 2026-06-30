/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuda.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "mha.h"

namespace mixed_xqa_cubin_loader {
#include <flashinfer/cubin_loader.h>
}

namespace {

static_assert(K_CACHE_ELEM_ENUM == 2 && V_CACHE_ELEM_ENUM == 3);
static_assert(HEAD_ELEMS == 128 && HEAD_GRP_SIZE == 4 && TOKENS_PER_PAGE == 16);
static_assert(!SPEC_DEC && !SLIDING_WINDOW && !LOW_PREC_OUTPUT);

constexpr char kCubinPath[] = "xqa/cubin/fp8_k_nvfp4_v_h128_g4_p16.cubin";
constexpr char kCubinSha[] = "1250d87fa1268a44730f3ef80c8996e1b677a3d8222582b46951cddb3ca42e70";

void checkCu(CUresult result, char const* operation) {
  if (result == CUDA_SUCCESS) {
    return;
  }
  char const* name = nullptr;
  char const* message = nullptr;
  cuGetErrorName(result, &name);
  cuGetErrorString(result, &message);
  throw std::runtime_error(std::string(operation) +
                           " failed: " + (name == nullptr ? "unknown" : name) + " (" +
                           (message == nullptr ? "no detail" : message) + ")");
}

struct CubinKernel {
  CUmodule module{};
  CUfunction function{};
  uint32_t dynamicSmemBytes{};
  CUcontext context{};
};

CubinKernel& getKernel() {
  // Module loading and the DtoH metadata read happen once during warmup.
  // Subsequent launches do not synchronize with the GPU.
  static std::mutex mutex;
  static std::unordered_map<CUcontext, std::unique_ptr<CubinKernel>> kernels;
  thread_local CUcontext cachedContext{};
  thread_local CubinKernel* cachedKernel{};

  CUcontext current{};
  checkCu(cuCtxGetCurrent(&current), "cuCtxGetCurrent");
  if (current == nullptr) {
    throw std::runtime_error("mixed XQA cubin requires a current CUDA context");
  }
  if (current == cachedContext) {
    return *cachedKernel;
  }

  std::lock_guard<std::mutex> lock(mutex);
  auto it = kernels.find(current);
  if (it == kernels.end()) {
    auto kernel = std::make_unique<CubinKernel>();
    kernel->context = current;

    std::string const image = mixed_xqa_cubin_loader::getCubin(kCubinPath, kCubinSha);
    if (image.empty()) {
      throw std::runtime_error("cannot load mixed XQA cubin");
    }
    checkCu(cuModuleLoadData(&kernel->module, image.data()), "cuModuleLoadData");
    checkCu(cuModuleGetFunction(&kernel->function, kernel->module, "kernel_mha"),
            "cuModuleGetFunction");

    CUdeviceptr smemSymbol{};
    size_t smemSymbolBytes{};
    checkCu(cuModuleGetGlobal(&smemSymbol, &smemSymbolBytes, kernel->module, "smemSize"),
            "cuModuleGetGlobal(smemSize)");
    if (smemSymbolBytes != sizeof(kernel->dynamicSmemBytes)) {
      throw std::runtime_error("invalid mixed XQA smemSize symbol");
    }
    checkCu(cuMemcpyDtoH(&kernel->dynamicSmemBytes, smemSymbol, sizeof(kernel->dynamicSmemBytes)),
            "cuMemcpyDtoH(smemSize)");
    checkCu(cuFuncSetAttribute(kernel->function, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                               static_cast<int>(kernel->dynamicSmemBytes)),
            "cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES)");
    it = kernels.emplace(current, std::move(kernel)).first;
  }
  cachedContext = current;
  cachedKernel = it->second.get();
  return *cachedKernel;
}

struct MixedCacheList {
  GMemKCacheHead* kCache;
  GMemVCacheHead* vCache;
  GMemVCacheHeadSf* vScaleCache;
  KVCachePageIndex const* pageTable;
  uint32_t const* seqLens;
  uint32_t maxPagesPerSeq;
};

static_assert(sizeof(MixedCacheList) == 48);

}  // namespace

void launchMHAFlashInfer(uint32_t multiProcessorCount, uint32_t nbKHeads, uint32_t slidingWinSize,
                         float qScale, float const* qScalePtr, OutputHead* output,
#if LOW_PREC_OUTPUT
                         float rcpOutScale,
#endif
                         InputHead const* q, float const* attentionSinks,
                         GMemKCacheHead* kCacheVLLM, GMemVCacheHead* vCacheVLLM,
#if ENABLE_4BIT_K_CACHE
                         GMemKCacheHeadSf* kSfCacheVLLM,
#endif
#if ENABLE_4BIT_V_CACHE
                         GMemVCacheHeadSf* vSfCacheVLLM,
#endif
                         KVCachePageIndex const* kvCachePageList, uint32_t maxSeqLen,
                         uint32_t const* seqLen, uint32_t batchSize, float kCacheScale,
                         float const* kScalePtr, float vCacheScale, float const* vScalePtr,
#if SPEC_DEC
                         uint32_t qSeqLen, uint32_t const* qCuSeqLens, MaskType const* mask,
#endif
                         uint32_t* semaphores, void* scratch, bool enable_pdl,
                         uint64_t k_stride_page, uint64_t k_stride_token, uint64_t k_stride_head,
                         uint64_t v_stride_page, uint64_t v_stride_token, uint64_t v_stride_head,
                         uint64_t k_sf_stride_page, uint64_t k_sf_stride_token,
                         uint64_t k_sf_stride_head, uint64_t v_sf_stride_page,
                         uint64_t v_sf_stride_token, uint64_t v_sf_stride_head,
                         cudaStream_t stream) {
  unused(slidingWinSize);
  unused(k_sf_stride_page);
  unused(k_sf_stride_token);
  unused(k_sf_stride_head);

  constexpr uint32_t kSequenceTile = 256;
  uint32_t const numSubsequences = std::min(
      std::max(1U, multiProcessorCount / (batchSize * nbKHeads)), divUp(maxSeqLen, kSequenceTile));
  uint32_t const maxPagesPerSeq = exactDiv(maxSeqLen, tokensPerPage);
  MixedCacheList cacheList{kCacheVLLM,      vCacheVLLM, vSfCacheVLLM,
                           kvCachePageList, seqLen,     maxPagesPerSeq};

  uint32_t const kContainersPerHead = validElemsPerHead / KCacheElemConverter::ElemsPerContainer;
  uint32_t const vContainersPerHead = validElemsPerHead / VCacheElemConverter::ElemsPerContainer;
  uint32_t const vScaleElementsPerHead = validElemsPerHead / VCacheElemConverter::QuantVectorSize;
  uint32_t kStridePage = k_stride_page / kContainersPerHead;
  uint32_t kStrideToken = k_stride_token / kContainersPerHead;
  uint32_t kStrideHead = k_stride_head / kContainersPerHead;
  uint32_t vStridePage = v_stride_page / vContainersPerHead;
  uint32_t vStrideToken = v_stride_token / vContainersPerHead;
  uint32_t vStrideHead = v_stride_head / vContainersPerHead;
  uint32_t vScaleStridePage = v_sf_stride_page / vScaleElementsPerHead;
  uint32_t vScaleStrideToken = v_sf_stride_token / vScaleElementsPerHead;
  uint32_t vScaleStrideHead = v_sf_stride_head / vScaleElementsPerHead;
  uint32_t zeroStride = 0;

  void* kernelArgs[] = {
      &nbKHeads,       &qScale,           &qScalePtr,         &output,           &q,
      &attentionSinks, &cacheList,        &batchSize,         &kCacheScale,      &kScalePtr,
      &vCacheScale,    &vScalePtr,        &kStridePage,       &kStrideToken,     &kStrideHead,
      &vStridePage,    &vStrideToken,     &vStrideHead,       &zeroStride,       &zeroStride,
      &zeroStride,     &vScaleStridePage, &vScaleStrideToken, &vScaleStrideHead, &semaphores,
      &scratch,
  };

  auto& kernel = getKernel();
  CUlaunchAttribute pdlAttribute{};
  pdlAttribute.id = CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION;
  pdlAttribute.value.programmaticStreamSerializationAllowed = enable_pdl;
  CUlaunchConfig config{};
  config.gridDimX = numSubsequences;
  config.gridDimY = nbKHeads;
  config.gridDimZ = batchSize;
  config.blockDimX = 128;
  config.blockDimY = 1;
  config.blockDimZ = 2;
  config.sharedMemBytes = kernel.dynamicSmemBytes;
  config.hStream = reinterpret_cast<CUstream>(stream);
  config.attrs = &pdlAttribute;
  config.numAttrs = 1;
  checkCu(cuLaunchKernelEx(&config, kernel.function, kernelArgs, nullptr),
          "cuLaunchKernelEx(mixed XQA)");
}

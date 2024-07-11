#pragma once

#include <cassert>
#include <cstdio>
#include <cstdlib>

#include <chrono>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "cutlass/numeric_types.h"
#include <cute/arch/cluster_sm90.hpp>
#include <cute/tensor.hpp>
#include <cutlass/cluster_launch.hpp>
#include <cutlass/cutlass.h>

#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/command_line.h"
#include "cutlass/util/helper_cuda.hpp"
#include "cutlass/util/print_error.hpp"

#include "cutlass/detail/layout.hpp"

#include "shared_storage.h"

template <class TensorS, class TensorD, class SmemLayoutS, class ThreadLayoutS,
          class SmemLayoutD, class ThreadLayoutD>
__global__ static void __launch_bounds__(256, 1)
    transposeKernelSmem(TensorS const S, TensorD const D,
                        SmemLayoutS const smemLayoutS, ThreadLayoutS const tS,
                        SmemLayoutD const smemLayoutD, ThreadLayoutD const tD) {
  using namespace cute;
  using Element = typename TensorS::value_type;

  // Use Shared Storage structure to allocate aligned SMEM addresses.
  extern __shared__ char shared_memory[];
  using SharedStorage = SharedStorageTranspose<Element, SmemLayoutD>;
  SharedStorage &shared_storage =
      *reinterpret_cast<SharedStorage *>(shared_memory);

  // two different views of smem
  Tensor sS = make_tensor(make_smem_ptr(shared_storage.smem.data()),
                          smemLayoutS); // (bM, bN)
  Tensor sD = make_tensor(make_smem_ptr(shared_storage.smem.data()),
                          smemLayoutD); // (bN, bM)

  Tensor gS = S(make_coord(_, _), blockIdx.x, blockIdx.y); // (bM, bN)
  Tensor gD = D(make_coord(_, _), blockIdx.y, blockIdx.x); // (bN, bM)

  Tensor tSgS = local_partition(gS, tS, threadIdx.x); // (ThrValM, ThrValN)
  Tensor tSsS = local_partition(sS, tS, threadIdx.x); // (ThrValM, ThrValN)
  Tensor tDgD = local_partition(gD, tD, threadIdx.x);
  Tensor tDsD = local_partition(sD, tD, threadIdx.x);

#ifdef LP_DEBUG
  if (thread0()) {
    print("sS: "); print(sS); print("\n");
    print("gS: "); print(gS); print("\n");
    print("tS: "); print(tS); print("\n");
    print("tSgS: "); print(tSgS); print("\n");
    print("tSsS: "); print(tSsS); print("\n");
    print("sD: "); print(sD); print("\n");
    print("gD: "); print(gD); print("\n");
    print("tD: "); print(tD); print("\n");
    print("tDgD: "); print(tDgD); print("\n");
    print("tDsD: "); print(tDsD); print("\n");

  }
#endif

  cute::copy(tSgS, tSsS); // LDGSTS

  cp_async_fence();
  cp_async_wait<0>();
  __syncthreads();

  cute::copy(tDsD, tDgD);
}

template <typename Element, bool isSwizzled = true> void transpose_smem(TransposeParams<Element> params) {

  using namespace cute;

  //
  // Make tensors
  //
  auto tensor_shape = make_shape(params.M, params.N);
  auto tensor_shape_trans = make_shape(params.N, params.M);
  auto gmemLayoutS = make_layout(tensor_shape, LayoutRight{});
  auto gmemLayoutD = make_layout(tensor_shape_trans, LayoutRight{});
  Tensor tensor_S = make_tensor(make_gmem_ptr(params.input), gmemLayoutS);
  Tensor tensor_D = make_tensor(make_gmem_ptr(params.output), gmemLayoutD);

  //
  // Tile tensors
  //

  using bM = Int<64>;
  using bN = Int<32>;

  auto block_shape = make_shape(bM{}, bN{});       // (bM, bN)
  auto block_shape_trans = make_shape(bN{}, bM{}); // (bN, bM)

  Tensor tiled_tensor_S =
      tiled_divide(tensor_S, block_shape); // ((bM, bN), m', n')
  Tensor tiled_tensor_D =
      tiled_divide(tensor_D, block_shape_trans); // ((bN, bM), n', m')

  auto tileShapeS = make_layout(block_shape, LayoutRight{});
  auto tileShapeD = make_layout(block_shape_trans, LayoutRight{});

  auto smemLayoutS = tileShapeS;
  auto smemLayoutD = composition(smemLayoutS, tileShapeD);
  auto smemLayoutS_swizzle = composition(Swizzle<5, 0, 5>{}, tileShapeS);
  auto smemLayoutD_swizzle = composition(smemLayoutS_swizzle, tileShapeD);

  auto threadLayoutS =
      make_layout(make_shape(Int<8>{}, Int<32>{}), LayoutRight{});
  auto threadLayoutD =
      make_layout(make_shape(Int<8>{}, Int<32>{}), LayoutRight{});

  size_t smem_size = int(
      sizeof(SharedStorageTranspose<Element, decltype(smemLayoutS_swizzle)>));

#ifdef LP_DEBUG
  if (thread0()) {
    print("block_shape: "); print(block_shape); print("\n");
    print("block_shape_trans: "); print(block_shape_trans); print("\n");
    print("tensor_S: "); print(tensor_S); print("\n");
    print("tensor_D: "); print(tensor_D); print("\n");
    print("tileShapeS: "); print(tileShapeS); print("\n");
    print("tileShapeD: "); print(tileShapeD); print("\n");
    print("tiled_tensor_S: "); print(tiled_tensor_S); print("\n");
    print("tiled_tensor_D: "); print(tiled_tensor_D); print("\n");
    print("smemLayoutS: "); print(smemLayoutS); print("\n");
    print("smemLayoutD: "); print(smemLayoutD); print("\n");
    print("smemLayoutS_swizzle: "); print(smemLayoutS_swizzle); print("\n");
    print("smemLayoutD_swizzle: "); print(smemLayoutD_swizzle); print("\n");
    print("threadLayoutS: "); print(threadLayoutS); print("\n");
    print("threadLayoutD: "); print(threadLayoutD); print("\n");
    print("smem_size: "); print(smem_size); print("\n");
  }
#endif
  //
  // Determine grid and block dimensions
  //

  dim3 gridDim(
      size<1>(tiled_tensor_S),
      size<2>(tiled_tensor_S)); // Grid shape corresponds to modes m' and n'
  dim3 blockDim(size(threadLayoutS)); // 256 threads

  if constexpr (isSwizzled) {
    transposeKernelSmem<<<gridDim, blockDim, smem_size>>>(
        tiled_tensor_S, tiled_tensor_D, smemLayoutS_swizzle, threadLayoutS,
        smemLayoutD_swizzle, threadLayoutD);
  } else {
    transposeKernelSmem<<<gridDim, blockDim, smem_size>>>(
        tiled_tensor_S, tiled_tensor_D, smemLayoutS, threadLayoutS,
        smemLayoutD, threadLayoutD);
  }
}

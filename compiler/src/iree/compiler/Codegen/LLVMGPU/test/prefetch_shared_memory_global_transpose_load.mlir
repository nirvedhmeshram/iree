// RUN: iree-opt -pass-pipeline="builtin.module(func.func(iree-llvmgpu-prefetch-shared-memory),cse,canonicalize)" %s | FileCheck %s

// Verify that amdgpu.global_transpose_load is recognized as a global memory
// read root by the prefetching pass and gets pipelined: the first load is
// hoisted into a prologue before the loop, and each subsequent load is
// overlapped with the compute from the previous iteration.

// CHECK-LABEL: @prefetch_global_transpose_load
// CHECK-SAME: (%[[SRC:.*]]: memref<128x8xbf16>,
func.func @prefetch_global_transpose_load(
    %src: memref<128x8xbf16>,
    %out: memref<f32>) {
  %cst_f32 = arith.constant 0.0 : f32
  %cst_bf16 = arith.constant 0.0 : bf16
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index
  %alloc = memref.alloc() : memref<8xbf16, #gpu.address_space<workgroup>>

  // Prologue: first global_transpose_load hoisted before the loop.
  // CHECK: %[[PRO_LOAD:.*]] = amdgpu.global_transpose_load %[[SRC]][%c0, %c0]
  // CHECK: vector.transfer_write %[[PRO_LOAD]], %[[ALLOC:.*]][%c0]

  // Loop peeled by 1 iteration (0..127 instead of 0..128).
  // CHECK: scf.for %[[IV:.*]] = %c0 to %c127
  %result = scf.for %k = %c0 to %c128 step %c1 iter_args(%acc = %cst_f32) -> f32 {
    // global_transpose_load (stream-copy read root)
    %tr = amdgpu.global_transpose_load %src[%k, %c0] : memref<128x8xbf16> -> vector<8xbf16>
    vector.transfer_write %tr, %alloc[%c0] {in_bounds = [true]}
        : vector<8xbf16>, memref<8xbf16, #gpu.address_space<workgroup>>
    // Compute from shared memory.
    %v = vector.transfer_read %alloc[%c0], %cst_bf16
        : memref<8xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
    %reduced = vector.reduction <add>, %v : vector<8xbf16> into bf16
    %ext = arith.extf %reduced : bf16 to f32
    %new_acc = arith.addf %acc, %ext : f32
    scf.yield %new_acc : f32
  }

  // Inside the pipelined loop:
  //   prefetch next iteration's load → barrier → compute from shared → barrier → write prefetched
  // CHECK:   %[[NEXT:.*]] = arith.addi %[[IV]], %c1
  // CHECK:   %[[KER_LOAD:.*]] = amdgpu.global_transpose_load %[[SRC]][%[[NEXT]], %c0]
  // CHECK:   gpu.barrier
  // CHECK:   vector.transfer_read %[[ALLOC]]
  // CHECK:   gpu.barrier
  // CHECK:   amdgpu.sched_barrier
  // CHECK:   vector.transfer_write %[[KER_LOAD]], %[[ALLOC]]

  // Epilogue: one final compute from shared (the last loaded tile).
  // CHECK: gpu.barrier
  // CHECK: vector.transfer_read %[[ALLOC]]

  memref.store %result, %out[] : memref<f32>
  return
}

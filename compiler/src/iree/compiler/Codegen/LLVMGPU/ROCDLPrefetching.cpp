// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "iree/compiler/Codegen/LLVMGPU/Utils/LLVMGPUUtils.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_ROCDLPREFETCHSHAREDMEMORYPASS
#include "iree/compiler/Codegen/LLVMGPU/Passes.h.inc"

namespace {

static void prefetchSharedMemory(FunctionOpInterface funcOp,
                                 unsigned numStages) {
  IRRewriter rewriter(funcOp.getContext());
  SmallVector<scf::ForOp> loops;
  // Collect only outermost loops (not nested inside other scf.for loops)
  funcOp.walk([&loops](scf::ForOp forOp) {
    // Check if this loop is nested inside another scf.for loop
    if (!forOp->getParentOfType<scf::ForOp>()) {
      loops.push_back(forOp);
    }
  });

  for (scf::ForOp forOp : loops) {
    // Collect imperfectly nested loops with chained iter_args.
    // The upstream coalesceLoops handles delinearization of IVs,
    // so operations between loops that use the IV will work correctly.
    SmallVector<scf::ForOp> nestedLoops;
    scf::ForOp currentLoop = forOp;

    while (currentLoop) {
      nestedLoops.push_back(currentLoop);

      Block &body = currentLoop.getRegion().front();

      // Look for the next nested loop
      scf::ForOp nextLoop = nullptr;
      for (Operation &op : body) {
        if (auto innerFor = dyn_cast<scf::ForOp>(&op)) {
          // Check if iter_args are properly chained
          if (currentLoop.getNumRegionIterArgs() ==
              innerFor.getNumRegionIterArgs() &&
              llvm::equal(currentLoop.getRegionIterArgs(),
                         innerFor.getInitArgs())) {
            nextLoop = innerFor;
          }
          break;
        }
      }

      currentLoop = nextLoop;
    }

    // Try to coalesce the nested loops if there are multiple.
    // The upstream coalesceLoops now supports imperfectly nested loops
    // by inserting delinearization at the start of the outermost loop.
    if (nestedLoops.size() > 1) {
      (void)coalesceLoops(rewriter, nestedLoops);
    }

    //FailureOr<scf::ForOp> newLoop =
    //    prefetchSharedMemoryCopy(rewriter, forOp, numStages);
    // The only possible failure is the analysis failure, which does not cause
    // the pass to fail. Therefore we discard any failures at this point.
    //(void)newLoop;
    break; // TODO: Fix nested loop handling.
  }
}

struct ROCDLPrefetchSharedMemoryPass final
    : impl::ROCDLPrefetchSharedMemoryPassBase<ROCDLPrefetchSharedMemoryPass> {
  using Base::Base;

  void runOnOperation() override {
    FunctionOpInterface funcOp = getOperation();
    prefetchSharedMemory(funcOp, numStages);
  }
};

} // namespace
} // namespace mlir::iree_compiler

// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/Util/Analysis/IntegerDivisibilityAnalysis.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"

#define DEBUG_TYPE "iree-codegen-rocdl-buffer-instructions-optimization"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_ROCDLBUFFERINSTRUCTIONSOPTIMIZATIONPASS
#include "iree/compiler/Codegen/LLVMGPU/ROCDLPasses.h.inc"

namespace {

Value createI1And(Location loc, ArrayRef<Value> values, OpBuilder &builder) {
  Value base = arith::IndexCastUIOp::create(builder, loc, builder.getI1Type(),
                                            values[0]);
  for (Value value : values.drop_front()) {
    Value rhs =
        arith::IndexCastUIOp::create(builder, loc, builder.getI1Type(), value);
    base = arith::AndIOp::create(builder, loc, base, rhs);
  }
  return base;
}
// Determine if the mask vector is all ones or all zeros and if so, then replace
// the masked transferReadOp with transferReadOp with no mask for when the mask
// is all ones and the padding values when the mask is all zeros. The pattern
// thus does the following optimization.
// clang-format off
// mask = vector.create_mask %0, ..., %n, %c8 : vector<1x ... x1x8xi1>
// %read = vector.transfer_read %memref, %mask : memref<..., amdgpu.raw_fat_buffer>
// becomes
// %padding = arith.constant dense<0> : vector<1x ... x1x8xbf16>
// %read = vector.transfer_read %memref : memref<..., amdgpu.raw_fat_buffer> // no mask!
// %masked_read
//   = arith.select %0 && ... && %n ? %read : %padding : index,vector<1x ... x1x8xbf16>
// clang-format on
// Note we currently dont support cases where muliple masks are ANDed or ORed
// together to form the final mask to a read but such support can be added where
// we track a set of valid masks and add that an AND or OR of valid masks is
// valid

void simplifyMaskOps(RewriterBase &rewriter, vector::CreateMaskOp maskOp,
                     DataFlowSolver &solver) {
  Location loc = maskOp.getLoc();

  SmallVector<vector::TransferReadOp> validReads;
  // First determine if the mask meets the criteria of being either all ones or
  // all empty.
  SmallVector<Value> ValuesToAnd;
  SmallVector<Value> maskIndices = maskOp.getOperands();
  ArrayRef<int64_t> maskShape = maskOp.getResult().getType().getShape();
  bool isValid = true;
  Value innermostNonConstantMaskIndex = nullptr;

  for (auto [idx, maskIndex] : llvm::enumerate(maskIndices)) {

    std::optional<int64_t> constantValue = getConstantIndex(maskIndex);
    if (constantValue) {
      if (maskShape[idx] != constantValue) {
        isValid = false;
        break;
      }
    } else {
      // For non-constant mask indices, we either need:
      // 1. The mask shape dimension to be 1 (will be added to ValuesToAnd)
      // 2. Or it's the innermost dimension (will be handled specially if stride
      // is divisible)
      bool isInnermostDim = (idx == maskIndices.size() - 1);
      if (maskShape[idx] == 1) {
        ValuesToAnd.push_back(maskIndex);
      } else if (isInnermostDim) {
        // Save this for later stride divisibility check
        innermostNonConstantMaskIndex = maskIndex;
      } else {
        isValid = false;
        break;
      }
    }
  }
  // Bail out if the mask doesnt meet the criteria or
  // is statically all 1's in which case we dont need
  // to do anything.
  if (!isValid || (ValuesToAnd.empty() && !innermostNonConstantMaskIndex)) {
    return;
  }

  for (Operation *user : maskOp.getResult().getUsers()) {
    auto readOp = dyn_cast<vector::TransferReadOp>(user);
    // Only TransferReadOps are supported.
    if (!readOp)
      continue;

    auto sourceType = dyn_cast<MemRefType>(readOp.getBase().getType());
    // only supported for fat raw buffers.
    if (!sourceType || !hasAMDGPUFatRawBufferAddressSpace(sourceType))
      continue;

    SmallVector<bool> inBounds = readOp.getInBoundsValues();
    // Only supported for reads that are fully in_bounds.
    if (inBounds.size() != sourceType.getRank() ||
        llvm::any_of(inBounds, [](bool inBound) { return !inBound; })) {
      continue;
    }

    rewriter.setInsertionPoint(readOp);

    Value selectValue = nullptr;

    // Check if we need to handle the innermost dimension specially
    if (innermostNonConstantMaskIndex) {
      int64_t innerDimIdx = sourceType.getRank() - 1;
      int64_t maskInnerDimSize = maskShape[innerDimIdx];

      // Use divisibility analysis on the mask index
      auto *lattice =
          solver.lookupState<IREE::Util::IntegerDivisibilityLattice>(
              innermostNonConstantMaskIndex);

      if (lattice && !lattice->getValue().isUninitialized()) {
        const auto &div = lattice->getValue().getValue();
        llvm::errs() << "Divisibility analysis for mask index:\n";
        llvm::errs() << "  udiv = " << div.udiv() << ", sdiv = " << div.sdiv()
                     << "\n";
        llvm::errs() << "  mask size = " << maskInnerDimSize << "\n";

        // Check if the mask index is divisible by the mask size
        if (div.udiv() % maskInnerDimSize == 0) {
          llvm::errs() << "  -> Divisible! Can optimize.\n";
        } else {
          llvm::errs() << "  -> Not divisible. Skipping optimization.\n";
          continue;
        }
      } else {
        llvm::errs() << "Divisibility analysis uninitialized for mask index. "
                        "Skipping.\n";
        continue;
      }

      // Create a compare: innerDimMaskIndex == maskInnerDimSize
      Value maskSizeConstant =
          arith::ConstantIndexOp::create(rewriter, loc, maskInnerDimSize);
      Value cmpResult = arith::CmpIOp::create(
          rewriter, loc, arith::CmpIPredicate::eq,
          innermostNonConstantMaskIndex, maskSizeConstant);

      // Start with this comparison
      if (ValuesToAnd.empty()) {
        selectValue = cmpResult;
      } else {
        // Combine with other mask conditions
        Value andValue = createI1And(loc, ValuesToAnd, rewriter);
        selectValue = arith::AndIOp::create(rewriter, loc, andValue, cmpResult);
      }
    } else {
      // No special innermost handling needed
      selectValue = createI1And(loc, ValuesToAnd, rewriter);
    }
    auto constantValue = vector::BroadcastOp::create(
        rewriter, loc, readOp.getVectorType(), readOp.getPadding());

    auto newReadOp = vector::TransferReadOp::create(
        rewriter, loc, readOp.getVectorType(), readOp.getBase(),
        readOp.getIndices(), readOp.getPadding(), ArrayRef<bool>{inBounds});
    auto selectOp = arith::SelectOp::create(rewriter, loc, selectValue,
                                            newReadOp, constantValue);
    rewriter.replaceAllUsesWith(readOp, selectOp);
  }
}

struct ROCDLBufferInstructionsOptimizationPass final
    : impl::ROCDLBufferInstructionsOptimizationPassBase<
          ROCDLBufferInstructionsOptimizationPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    FunctionOpInterface funcOp = getOperation();

    // Setup divisibility analysis
    DataFlowSolver solver;
    solver.load<dataflow::DeadCodeAnalysis>();
    solver.load<IREE::Util::IntegerDivisibilityAnalysis>();
    solver.load<dataflow::SparseConstantPropagation>();
    if (failed(solver.initializeAndRun(funcOp))) {
      return signalPassFailure();
    }

    SmallVector<vector::CreateMaskOp> maskOps;
    funcOp.walk(
        [&](vector::CreateMaskOp maskOp) { maskOps.push_back(maskOp); });

    IRRewriter rewriter(context);
    for (vector::CreateMaskOp maskOp : maskOps) {
      simplifyMaskOps(rewriter, maskOp, solver);
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler

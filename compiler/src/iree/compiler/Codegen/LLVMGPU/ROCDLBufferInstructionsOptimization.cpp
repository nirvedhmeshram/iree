// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"

#define DEBUG_TYPE "iree-codegen-rocdl-buffer-instructions-optimization"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_ROCDLBUFFERINSTRUCTIONSOPTIMIZATIONPASS
#include "iree/compiler/Codegen/LLVMGPU/ROCDLPasses.h.inc"

static constexpr char kPeeledLoopLabel[] = "__peeled_loop__";
static constexpr char kPartialIterationLabel[] = "__partial_iteration__";

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

struct SimplifyMaskVectorTransferRead final
    : OpRewritePattern<vector::CreateMaskOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::CreateMaskOp maskOp,
                                PatternRewriter &rewriter) const override {
    Location loc = maskOp.getLoc();

    SmallVector<vector::TransferReadOp> validReads;
    // First determine if the mask meets the criteria of being either all ones
    // or all empty.
    SmallVector<Value> ValuesToAnd;
    SmallVector<Value> maskIndices = maskOp.getOperands();
    ArrayRef<int64_t> maskShape = maskOp.getResult().getType().getShape();
    bool isValid = true;
    for (auto [idx, maskIndex] : llvm::enumerate(maskIndices)) {

      std::optional<int64_t> constantValue = getConstantIndex(maskIndex);
      if (constantValue) {
        if (maskShape[idx] != constantValue) {
          isValid = false;
          break;
        }
      } else {
        if (maskShape[idx] != 1) {
          isValid = false;
          break;
        }
        ValuesToAnd.push_back(maskIndex);
      }
    }
    // Bail out if the mask doesnt meet the criteria or
    // is statically all 1's in which case we dont need
    // to do anything.
    if (!isValid || ValuesToAnd.empty()) {
      return failure();
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
      Value selectValue = createI1And(loc, ValuesToAnd, rewriter);
      auto constantValue = vector::BroadcastOp::create(
          rewriter, loc, readOp.getVectorType(), readOp.getPadding());

      auto newReadOp = vector::TransferReadOp::create(
          rewriter, loc, readOp.getVectorType(), readOp.getBase(),
          readOp.getIndices(), readOp.getPadding(), ArrayRef<bool>{inBounds});
      auto selectOp = arith::SelectOp::create(rewriter, loc, selectValue,
                                              newReadOp, constantValue);
      rewriter.replaceAllUsesWith(readOp, selectOp);
    }
    return success();
  }
};

struct PeelPartialMaskLoops final
    : OpRewritePattern<vector::CreateMaskOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::CreateMaskOp maskOp,
                                PatternRewriter &rewriter) const override {
  Operation* parentOp;
  Operation* currOp = maskOp;
  while ((parentOp = currOp->getParentOfType<scf::ForOp>())){
    currOp = parentOp;
  }
  currOp->dump();
  llvm::outs()<<"Here 1\n";
  llvm::outs().flush();
  if(!isa<scf::ForOp>(currOp)){
    return failure();
  }
  llvm::outs()<<"Here 2\n";
  llvm::outs().flush();
  scf::ForOp partialIteration;
  scf::ForOp forOp = cast<scf::ForOp>(currOp);
  // Do not peel already peeled loops.
  if (forOp->hasAttr(kPeeledLoopLabel))
      return failure();
  if(failed(peelForLoopAndSimplifyBounds(rewriter,forOp,partialIteration))){
    return failure();
  }
  llvm::outs()<<"Here 3\n";
  llvm::outs().flush();
    // Apply label, so that the same loop is not rewritten a second time.
    rewriter.modifyOpInPlace(partialIteration, [&]() {
      partialIteration->setAttr(kPeeledLoopLabel, rewriter.getUnitAttr());
      partialIteration->setAttr(kPartialIterationLabel, rewriter.getUnitAttr());
    });
    rewriter.modifyOpInPlace(forOp, [&]() {
      forOp->setAttr(kPeeledLoopLabel, rewriter.getUnitAttr());
    });
  llvm::outs()<<"Here 4\n";
  llvm::outs().flush();
  return success();



  }
};

struct ROCDLBufferInstructionsOptimizationPass final
    : impl::ROCDLBufferInstructionsOptimizationPassBase<
          ROCDLBufferInstructionsOptimizationPass> {
  void runOnOperation() override {
    FunctionOpInterface funcOp = getOperation();
    MLIRContext *context = &getContext();
    {
    RewritePatternSet patterns(context);
    patterns.add<SimplifyMaskVectorTransferRead>(context);
    (void)applyPatternsGreedily(funcOp, std::move(patterns));
    }
    {
    RewritePatternSet patterns(context);
    scf::ForOp::getCanonicalizationPatterns(patterns, context);
    patterns.add<PeelPartialMaskLoops>(context);
    (void)applyPatternsGreedily(funcOp, std::move(patterns));
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler

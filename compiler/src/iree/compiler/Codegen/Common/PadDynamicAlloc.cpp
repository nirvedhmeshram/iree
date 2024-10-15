// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_PADDYNAMICALLOCPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

/// If a value is defined by `%dim = affine_max(0, %src)` kind of op return
/// `%src` otherwise return `%dim`.
/// This is useful to handle common pattern generated by bufferization to
/// compute alloc sizes.
static Value skipAffineMaxZero(Value dim) {
  auto affineMax = dim.getDefiningOp<affine::AffineMaxOp>();
  if (!affineMax)
    return dim;
  for (AffineExpr expr : affineMax.getMap().getResults()) {
    if (auto cst = dyn_cast<AffineConstantExpr>(expr)) {
      if (cst.getValue() == 0)
        continue;
    } else if (auto symExpr = dyn_cast<AffineSymbolExpr>(expr)) {
      if (symExpr.getPosition() == 0)
        continue;
    }
    return dim;
  }
  return *affineMax.getSymbolOperands().begin();
}

template <typename OpTy>
static LogicalResult padAlloc(MLIRContext *context, OpTy allocOp) {
  IRRewriter rewriter(context);
  rewriter.setInsertionPoint(allocOp);
  SmallVector<int64_t> shape = llvm::to_vector(allocOp.getType().getShape());
  SmallVector<OpFoldResult> sizes;
  size_t dynamicDimIdx = 0;
  for (int64_t &dimSize : shape) {
    if (!ShapedType::isDynamic(dimSize)) {
      sizes.push_back(rewriter.getIndexAttr(dimSize));
      continue;
    }
    Value dim = allocOp.getDynamicSizes()[dynamicDimIdx++];
    dim = skipAffineMaxZero(dim);
    auto ub = ValueBoundsConstraintSet::computeConstantBound(
        presburger::BoundType::UB, {dim, /*dim=*/std::nullopt},
        /*stopCondition=*/nullptr, /*closedUB=*/true);
    if (failed(ub)) {
      return allocOp.emitOpError(
          "unexpected allocation without upper bound shapes");
    }
    dimSize = *ub;
    sizes.push_back(dim);
  }
  if (dynamicDimIdx == 0)
    return success();
  Type elType = allocOp.getType().getElementType();
  MemRefType allocType = MemRefType::get(shape, elType, AffineMap(),
                                         allocOp.getType().getMemorySpace());
  Location loc = allocOp.getLoc();
  Value paddedAlloc = rewriter.create<OpTy>(loc, allocType);
  SmallVector<OpFoldResult> offsets(shape.size(), rewriter.getIndexAttr(0));
  SmallVector<OpFoldResult> strides(shape.size(), rewriter.getIndexAttr(1));
  Value subview = rewriter.create<memref::SubViewOp>(loc, paddedAlloc, offsets,
                                                     sizes, strides);
  replaceMemrefUsesAndPropagateType(rewriter, loc, allocOp, subview);
  rewriter.eraseOp(allocOp);
  return success();
}

namespace {

struct PadDynamicAllocPass final
    : impl::PadDynamicAllocPassBase<PadDynamicAllocPass> {
  void runOnOperation() override {
    auto funcOp = getOperation();
    MLIRContext *context = &getContext();
    SmallVector<memref::AllocOp> sharedMemAllocs;
    // Collect all the alloc operations.
    funcOp.walk(
        [&](memref::AllocOp allocOp) { sharedMemAllocs.push_back(allocOp); });
    for (memref::AllocOp alloc : sharedMemAllocs) {
      if (failed(padAlloc<memref::AllocOp>(context, alloc)))
        return signalPassFailure();
    }

    SmallVector<memref::AllocaOp> privateMemAllocas;
    // Collect all the alloc operations.
    funcOp.walk([&](memref::AllocaOp allocaOp) {
      privateMemAllocas.push_back(allocaOp);
    });
    for (memref::AllocaOp alloca : privateMemAllocas) {
      if (failed(padAlloc<memref::AllocaOp>(context, alloca)))
        return signalPassFailure();
    }
  }
};
} // namespace
} // namespace mlir::iree_compiler

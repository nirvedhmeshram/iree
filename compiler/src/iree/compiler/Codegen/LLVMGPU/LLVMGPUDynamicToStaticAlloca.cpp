// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMGPU/TransformExtensions/LLVMGPUExtensions.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "llvm/Support/Debug.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/TransformOps/GPUTransformOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "iree-llvmgpu-dynamic-to-static-alloca"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_LLVMGPUDYNAMICTOSTATICALLOCAPASS
#include "iree/compiler/Codegen/LLVMGPU/Passes.h.inc"

namespace {

struct LLVMGPUDynamicToStaticAllocaPass final
    : impl::LLVMGPUDynamicToStaticAllocaPassBase<
          LLVMGPUDynamicToStaticAllocaPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, gpu::GPUDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    IRRewriter rewriter(context);

    FunctionOpInterface funcOp = getOperation();

    auto replaceAlloca = [&](memref::AllocaOp alloca) {
      int64_t destRank = alloca.getType().getRank();
      SmallVector<int64_t> staticShape;
      for (int64_t dim = 0; dim < destRank; ++dim) {
        FailureOr<int64_t> maybeDimBound =
            ValueBoundsConstraintSet::computeConstantBound(
                presburger::BoundType::UB, {alloca.getResult(), dim},
                /*stopCondition=*/nullptr, /*closedUB=*/true);
        if (failed(maybeDimBound)) {
          return;
        }
        staticShape.push_back(maybeDimBound.value());
      }
      SmallVector<OpFoldResult> staticSizes =
          getAsIndexOpFoldResult(context, staticShape);
      rewriter.setInsertionPoint(alloca);
      auto staticAlloca = rewriter.create<memref::AllocaOp>(
          alloca->getLoc(), staticSizes, alloca.getType().getElementType(),
          alloca.getType().getMemorySpace());
      // SmallVector<OpFoldResult> offsets(destRank, rewriter.getIndexAttr(0));
      SmallVector<OpFoldResult> strides(destRank, rewriter.getIndexAttr(1));
      rewriter.replaceOpWithNewOp<memref::ReinterpretCastOp>(
          alloca, alloca.getType(), staticAlloca, rewriter.getIndexAttr(0),
          alloca.getMixedSizes(), strides);
    };

    SmallVector<memref::AllocaOp> dynamicAllocas;
    funcOp->walk([&](memref::AllocaOp allocaOp) {
      MemRefType allocaType = allocaOp.getType();
      if (allocaType.hasStaticShape()) {
        return;
      }
      dynamicAllocas.push_back(allocaOp);
    });

    for (memref::AllocaOp alloca : dynamicAllocas) {
      replaceAlloca(alloca);
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler

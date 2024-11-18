// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUDialect.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUEnums.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUInterfaces.h"
#include "iree/compiler/Codegen/Utils/LinalgOpInfo.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/Passes.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_GPUPADOPERANDSPASS
#include "iree/compiler/Codegen/Common/GPU/Passes.h.inc"

namespace {


static LogicalResult padOrPromoteContractionLikeOp(RewriterBase &rewriter,
                                                   linalg::LinalgOp linalgOp,
                                                   ArrayRef<int64_t> padding) {
  //Location loc = linalgOp.getLoc();

  SmallVector<int64_t> paddingDims =
      llvm::to_vector(llvm::seq<int64_t>(0, linalgOp.getNumLoops()));
  SmallVector<bool> nofoldFlags(linalgOp.getNumDpsInputs(), /*nofold=*/false);
  SmallVector<Attribute> paddingValueAttributes;
  for (auto &operand : linalgOp->getOpOperands()) {
    auto elemType = getElementTypeOrSelf(operand.get().getType());
    paddingValueAttributes.push_back(rewriter.getZeroAttr(elemType));
  }

  auto options =
      linalg::LinalgPaddingOptions()
          .setPaddingDimensions(paddingDims)
          .setPaddingValues(paddingValueAttributes)
          .setPadToMultipleOf(padding)
          .setNofoldFlags(nofoldFlags)
          .setCopyBackOp(linalg::LinalgPaddingOptions::CopyBackOp::None);

  linalg::LinalgOp paddedOp;
  SmallVector<Value> newResults;
  SmallVector<tensor::PadOp> padOps;
  if (failed(rewriteAsPaddedOp(rewriter, linalgOp, options, paddedOp,
                               newResults, padOps))) {
    return rewriter.notifyMatchFailure(linalgOp,
                                       "failed to pad contraction op");
  }
  rewriter.replaceOp(linalgOp, newResults.front());
  return success();
}

struct GPUPadOperandsPass final
    : impl::GPUPadOperandsPassBase<GPUPadOperandsPass> {
  void runOnOperation() override {
    FunctionOpInterface funcOp = getOperation();

    IRRewriter rewriter(funcOp);
    funcOp.walk([&](linalg::LinalgOp op) {
      auto loweringConfig =
          getLoweringConfig<IREE::GPU::LoweringConfigAttr>(op);
      if (!loweringConfig) {
        return;
      }

      std::optional<SmallVector<int64_t>> paddingTileSizes =
          loweringConfig.getPaddingList();
      if (!paddingTileSizes) {
        return;
      }

      rewriter.setInsertionPoint(op);
      for (auto operand : paddingTileSizes.value()) {
        llvm::outs()<<operand<<" ";
      }
      llvm::outs().flush();

      if (failed(padOrPromoteContractionLikeOp(rewriter, op,
                                               paddingTileSizes.value()))) {
        return signalPassFailure();
      }
    });
  }
};

} // namespace
} // namespace mlir::iree_compiler

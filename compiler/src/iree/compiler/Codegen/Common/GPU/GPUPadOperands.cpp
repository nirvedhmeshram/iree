// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/GPULoweringConfigUtils.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_GPUPADOPERANDSPASS
#include "iree/compiler/Codegen/Common/GPU/Passes.h.inc"


struct SimplifyFillPadPattern final : OpRewritePattern<tensor::PadOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tensor::PadOp padOp,
                                PatternRewriter &rewriter) const override {
    Operation *currentOp = padOp.getSource().getDefiningOp();
    auto maybeExtractSlice =
        dyn_cast_or_null<tensor::ExtractSliceOp>(currentOp);
    while (currentOp && maybeExtractSlice) {
      currentOp = maybeExtractSlice.getSource().getDefiningOp();
      maybeExtractSlice = dyn_cast_or_null<tensor::ExtractSliceOp>(currentOp);
    }
    auto fillOp = dyn_cast_or_null<linalg::FillOp>(currentOp);
    if (!fillOp) {
      return rewriter.notifyMatchFailure(
          padOp, "not coming from a linalg.fill op via tensor.extract_slice*");
    }

    Value padValue = padOp.getConstantPaddingValue();
    RankedTensorType resultType = padOp.getResultType();
    if (!padValue ||
        getAsOpFoldResult(padValue) !=
            getAsOpFoldResult(fillOp.getDpsInputOperand(0)->get())) {
      return rewriter.notifyMatchFailure(
          padOp, "not a constant value matching the fill value");
    }

    Location loc = padOp.getLoc();
    auto emptyOp = rewriter.create<tensor::EmptyOp>(
        loc, tensor::getMixedSizes(rewriter, loc, padOp),
        resultType.getElementType());
    rewriter.replaceOpWithNewOp<linalg::FillOp>(padOp, padValue,
                                                emptyOp.getResult());

    return success();
  }                     
};

namespace {

static LogicalResult padLinalgOpToStaticSizes(RewriterBase &rewriter,
                                              linalg::LinalgOp linalgOp,
                                              ArrayRef<int64_t> padding) {
  SmallVector<int64_t> paddingDims =
      llvm::to_vector(llvm::seq<int64_t>(0, linalgOp.getNumLoops()));
  SmallVector<bool> nofoldFlags(linalgOp.getNumDpsInputs(), /*nofold=*/false);
  SmallVector<Attribute> paddingValueAttributes;
  for (auto &operand : linalgOp->getOpOperands()) {
    Type elemType = getElementTypeOrSelf(operand.get().getType());
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
          getPaddingList(loweringConfig);
      if (!paddingTileSizes) {
        return;
      }

      rewriter.setInsertionPoint(op);
      if (failed(padLinalgOpToStaticSizes(rewriter, op,
                                          paddingTileSizes.value()))) {
        return signalPassFailure();
      }
    });

  // Cleanup patterns.
  {
  MLIRContext *context = &getContext();
  RewritePatternSet cleanupPatterns(context);
  cleanupPatterns.add<SimplifyFillPadPattern>(context);
  if (failed(applyPatternsGreedily(funcOp, std::move(cleanupPatterns)))) {
    return signalPassFailure();
  }
  }

  }
};

} // namespace
} // namespace mlir::iree_compiler

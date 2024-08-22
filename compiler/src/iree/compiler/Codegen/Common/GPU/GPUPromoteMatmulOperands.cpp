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

#define GEN_PASS_DEF_GPUPROMOTEMATMULOPERANDSPASS
#include "iree/compiler/Codegen/Common/GPU/Passes.h.inc"

namespace {

/// Inserts a `linalg.copy` directly before the given operation on the
/// specified operand, for example with operand index = 1:
///
///   linalg.matmul ins(%0, %1)
///
/// becomes
///
///   %empty = tensor.empty()
///   %copy = linalg.copy %1 to %empty {
///     lowering_config = #iree_gpu.derived_thread_config}
///   linalg.matmul ins(%0, %copy)
///
/// If the producer is already a tilable op, the producer is just annotated with
/// #iree_gpu.derived_thread_config to indicate that it should be distributed
/// to threads independently of the matmul.
void promoteOperand(OpBuilder &builder, Operation *op, unsigned index) {
  Value operand = op->getOperand(index);

  if (auto producer = operand.getDefiningOp<TilingInterface>()) {
    // Skip promotion of fills.
    if (isa<linalg::FillOp>(producer)) {
      return;
    }
    if (auto generic = dyn_cast<linalg::GenericOp>(&*producer)) {
      if (linalg::isaFillOpInterface(generic)) {
        return;
      }
    }
    setLoweringConfig(producer, IREE::GPU::DerivedThreadConfigAttr::get(
                                    builder.getContext()));
    return;
  }

  auto tensorType = dyn_cast<RankedTensorType>(operand.getType());
  if (!tensorType) {
    return;
  }

  SmallVector<OpFoldResult> mixedSizes =
      tensor::getMixedSizes(builder, op->getLoc(), operand);
  Value empty = builder.create<tensor::EmptyOp>(op->getLoc(), mixedSizes,
                                                tensorType.getElementType());
  auto copy = builder.create<linalg::CopyOp>(op->getLoc(), operand, empty);
  setLoweringConfig(
      copy, IREE::GPU::DerivedThreadConfigAttr::get(builder.getContext()));
  op->setOperand(index, copy.getResult(0));
}
static bool isNonMatvecContraction(linalg::LinalgOp linalgOp) {
  SmallVector<int64_t, 4> bounds = linalgOp.getStaticLoopRanges();
  FailureOr<mlir::linalg::ContractionDimensions> contractionDims =
      mlir::linalg::inferContractionDims(linalgOp);
  if (failed(contractionDims)) {
    return false;
  }

  if (contractionDims->k.size() < 1 || contractionDims->m.size() < 1 ||
      contractionDims->n.size() < 1) {
    return false;
  }

  auto getElementCount = [&](ArrayRef<unsigned> dims) {
    int64_t acc = 1;
    for (auto mDim : dims) {
      int64_t size = bounds[mDim];
      if (ShapedType::isDynamic(size)) {
        return size;
      }
      acc *= size;
    }
    return acc;
  };
  return getElementCount(contractionDims->m) != 1 &&
         getElementCount(contractionDims->n) != 1;
}

/// Creates a `linalg.copy` on the given tensor value and sets the lowering
/// configuration for the copy to `#iree_gpu.derived_thread_config`.
static Value promoteValue(OpBuilder &builder, Location loc, Value v) {
  auto tensorType = cast<RankedTensorType>(v.getType());
  SmallVector<OpFoldResult> mixedSizes = tensor::getMixedSizes(builder, loc, v);
  Value empty = builder.create<tensor::EmptyOp>(loc, mixedSizes,
                                                tensorType.getElementType());
  auto copy = builder.create<linalg::CopyOp>(loc, v, empty);
  setLoweringConfig(
      copy, IREE::GPU::DerivedThreadConfigAttr::get(builder.getContext()));
  return copy.getResult(0);
}

/// Inserts a `linalg.copy` directly before the given operation on the
/// specified operand, for example with operand index = 1:
///
///   linalg.matmul ins(%0, %1)
///
/// becomes
///
///   %empty = tensor.empty()
///   %copy = linalg.copy %1 to %empty {
///     lowering_config = #iree_gpu.derived_thread_config}
///   linalg.matmul ins(%0, %copy)
///
/// If the producer is already a tilable op, the producer is just annotated with
/// #iree_gpu.derived_thread_config to indicate that it should be distributed
/// to threads independently of the matmul.
static LogicalResult padOrPromoteContractionLikeOp(RewriterBase &rewriter,
                                                   linalg::LinalgOp linalgOp,
                                                   ArrayRef<int64_t> padding,
                                                   bool promoteC) {
  Location loc = linalgOp.getLoc();

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

  {
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(paddedOp);
    Value lhs = promoteValue(rewriter, loc, paddedOp->getOperand(0));
    paddedOp->setOperand(0, lhs);
    Value rhs = promoteValue(rewriter, loc, paddedOp->getOperand(1));
    paddedOp->setOperand(1, rhs);
  }

  Value replacement = newResults.front();
  auto extractSlice = replacement.getDefiningOp<tensor::ExtractSliceOp>();
  if (!extractSlice) {
    return failure();
  }

  bool canSkipPromotion =
      extractSlice.getSourceType() == extractSlice.getResultType() &&
      extractSlice.getSourceType().hasStaticShape();
  if (!canSkipPromotion || promoteC) {
    OpBuilder::InsertionGuard g(rewriter);
    Value valToMakeShared =
        extractSlice ? extractSlice.getSource() : replacement;
    rewriter.setInsertionPointAfterValue(valToMakeShared);
    auto tensorType = cast<RankedTensorType>(valToMakeShared.getType());
    SmallVector<Value> dynamicSizes;
    for (auto [idx, size] : llvm::enumerate(tensorType.getShape())) {
      if (ShapedType::isDynamic(size)) {
        dynamicSizes.push_back(
            rewriter.create<tensor::DimOp>(loc, valToMakeShared, idx));
      }
    }
    Attribute addressSpace = gpu::AddressSpaceAttr::get(
        rewriter.getContext(), gpu::GPUDialect::getWorkgroupAddressSpace());
    auto alloc = rewriter.create<bufferization::AllocTensorOp>(loc, tensorType,
                                                               dynamicSizes);
    alloc.setMemorySpaceAttr(addressSpace);
    Value copy =
        rewriter.create<linalg::CopyOp>(loc, valToMakeShared, alloc.getResult())
            .getResult(0);
    if (extractSlice) {
      extractSlice.getSourceMutable().assign(copy);
    } else {
      replacement = copy;
    }
    rewriter.setInsertionPointAfterValue(replacement);
    replacement = promoteValue(rewriter, loc, replacement);
  }

  rewriter.replaceOp(linalgOp, replacement);
  return success();
}

struct GPUPromoteMatmulOperandsPass final
    : impl::GPUPromoteMatmulOperandsPassBase<GPUPromoteMatmulOperandsPass> {
  void runOnOperation() override {
    FunctionOpInterface funcOp = getOperation();

    IRRewriter rewriter(funcOp);
    SmallVector<linalg::LinalgOp> promotionTargets;
    funcOp.walk([&](linalg::LinalgOp linalgOp) {
      if (isNonMatvecContraction(linalgOp)) {
        promotionTargets.push_back(linalgOp);
      }
    });

    for (auto linalgOp : promotionTargets) {
      rewriter.setInsertionPoint(linalgOp);

      auto loweringConfig =
          getLoweringConfig<IREE::GPU::LoweringConfigAttr>(linalgOp);

      // Skip operations without lowering configs.
      if (!loweringConfig) {
        continue;
      }

      SmallVector<int64_t> paddingTileSizes(linalgOp.getNumLoops(), 0);

      for (auto [i, size] :
           llvm::enumerate(loweringConfig.getWorkgroupTileSizes())) {
        paddingTileSizes[i] = size;
      }

      int64_t reduction =
          llvm::to_underlying(IREE::GPU::TilingLevel::Reduction);
      if (loweringConfig.hasTilingLevel(reduction)) {
        int64_t innerKDim = -1;
        int64_t kPackFactor = 1;
        if (IREE::GPU::MmaInterfaceAttr mma = loweringConfig.getMmaKind()) {
          linalg::ContractionDimensions contractionDims =
              *linalg::inferContractionDims(linalgOp);
          assert(!contractionDims.k.empty());
          innerKDim = contractionDims.k.back();
          kPackFactor = std::get<2>(mma.getMNKShape());
        }
        for (auto [i, size] :
             llvm::enumerate(loweringConfig.getStaticTilingLevelSizes(
                 reduction, linalgOp))) {
          if (!ShapedType::isDynamic(size) && size > 0) {
            paddingTileSizes[i] = i == innerKDim ? size * kPackFactor : size;
          }
        }
      }

      bool promoteC = loweringConfig.getAttributes().contains("promote_c");
      if (failed(padOrPromoteContractionLikeOp(rewriter, linalgOp,
                                               paddingTileSizes, promoteC))) {
        return signalPassFailure();
      }
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler

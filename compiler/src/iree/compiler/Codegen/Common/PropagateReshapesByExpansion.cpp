// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_PROPAGATERESHAPESBYEXPANSIONPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

/// Check if the users of the exapnded scf.forall destination can be updated to
/// account for the expand. If not we bail out. There are two suppoerted users
/// which are extract_slice -> expand_shape with the same exact reassociation map as the collapse op to be hoisted out.
/// or the parallel_insert_slice on which this pattern is rooted on.
static LogicalResult
verifyandCollectExpandableUsers(Value insertDest, SmallVector<ReassociationIndices> reIndices,
                                SmallVector<Operation *> &expandableUsers) {
  for (auto user : insertDest.getUsers()){
    if (auto extractSliceOp = dyn_cast<tensor::ExtractSliceOp>(user)) {
      auto expandShapeOp =
          dyn_cast<tensor::ExpandShapeOp>(*extractSliceOp->getUsers().begin());
      if(!expandShapeOp)
        return failure();
      auto expandReIndices = expandShapeOp.getReassociationIndices();
      if(reIndices != expandReIndices){
        return failure();
      }
      expandableUsers.push_back(user);
    }
    // We have already verified this op so we can just add it to exapndable users.
    else if(auto parallelInsertOp = dyn_cast<tensor::ParallelInsertSliceOp>(user)) {
      expandableUsers.push_back(user);
    }
    else
      return failure();
    
  }
  return success();
}

static void expandVerifiedExpandableUsers(
    MLIRContext *ctx, PatternRewriter &rewriter,
    SmallVector<Operation *> expandableUsers,
    SmallVector<SmallVector<OpFoldResult>> expandedDimsList) {
  for (auto user : expandableUsers) {
    auto loc = user->getLoc();
    if (auto extractSliceOp = dyn_cast<tensor::ExtractSliceOp>(user)) {
      rewriter.setInsertionPoint(extractSliceOp);
      auto expandShapeOp =
          dyn_cast<tensor::ExpandShapeOp>(*extractSliceOp->getUsers().begin());
      auto resultType = expandShapeOp.getResultType();
      auto mixedOffsets = extractSliceOp.getMixedOffsets();
      SmallVector<OpFoldResult> newOffsets;
      for (auto [index, offset] : llvm::enumerate(mixedOffsets)) {
        SmallVector<OpFoldResult> delinearizedOffsets =
            rewriter
                .create<affine::AffineDelinearizeIndexOp>(
                    loc, getValueOrCreateConstantIndexOp(rewriter, loc, offset),
                    expandedDimsList[index])
                ->getResults();
        newOffsets.push_back(delinearizedOffsets[0]);
        // We could use the inner delinearizedOffset  but since we have an
        // assumption that the slices are disjoint using zero offset for inner offsets reduces some
        // indexing math.
        for (int i = 1; i < expandedDimsList[index].size(); i++)
          newOffsets.push_back(getAsIndexOpFoldResult(ctx, 0));
      }
      SmallVector<OpFoldResult> newStrides(resultType.getRank(),
                                           rewriter.getIndexAttr(1));
      ArrayRef<int64_t> newShape = resultType.getShape();
      SmallVector<OpFoldResult> newSizes;
      for (auto size : newShape) {
        newSizes.push_back(getAsIndexOpFoldResult(ctx, size));
      }

      SmallVector<ReassociationIndices> flattenReassociations;
      auto reInds = expandShapeOp.getReassociationIndices();
      for(auto inds : reInds){
        for (auto i : inds){
          flattenReassociations.push_back({i});
      }
      }
      rewriter.replaceOpWithNewOp<tensor::ExtractSliceOp>(
              extractSliceOp, resultType, extractSliceOp.getSource(),
              newOffsets, newSizes, newStrides);
      rewriter.setInsertionPoint(expandShapeOp);
      rewriter.replaceOpWithNewOp<tensor::ExpandShapeOp>(
        expandShapeOp, resultType,
        expandShapeOp.getSrc(),flattenReassociations);
    }
    else if (auto parallelInsertOp = dyn_cast<tensor::ParallelInsertSliceOp>(user)) {
      auto collapseShapeOp =
          parallelInsertOp.getSource().getDefiningOp<tensor::CollapseShapeOp>();
      rewriter.setInsertionPoint(parallelInsertOp->getParentOp());
      auto resultType = collapseShapeOp.getSrcType();
      auto mixedOffsets = parallelInsertOp.getMixedOffsets();
      SmallVector<OpFoldResult> newOffsets;
      for (auto [index, offset] : llvm::enumerate(mixedOffsets)) {
        SmallVector<OpFoldResult> delinearizedOffsets =
            rewriter
                .create<affine::AffineDelinearizeIndexOp>(
                    loc, getValueOrCreateConstantIndexOp(rewriter, loc, offset),
                    expandedDimsList[index])
                ->getResults();
        newOffsets.push_back(delinearizedOffsets[0]);
        // We could use the inner delinearizedOffset  but since we have an
        // assumption that the slices are disjoint using zero reduces some
        // indexing math.
        for (int i = 1; i < expandedDimsList[index].size(); i++)
          newOffsets.push_back(getAsIndexOpFoldResult(ctx, 0));
      }
      SmallVector<OpFoldResult> newStrides(resultType.getRank(),
                                           rewriter.getIndexAttr(1));
      ArrayRef<int64_t> newShape = resultType.getShape();
      SmallVector<OpFoldResult> newSizes;
      for (auto size : newShape) {
        newSizes.push_back(getAsIndexOpFoldResult(ctx, size));
      }

      SmallVector<ReassociationIndices> flattenReassociations;
      auto reInds = collapseShapeOp.getReassociationIndices();
      for(auto inds : reInds){
        for (auto i : inds){
          flattenReassociations.push_back({i});
        }

      }
      rewriter.setInsertionPoint(collapseShapeOp);
          auto newCollapseOp = rewriter.replaceOpWithNewOp<tensor::CollapseShapeOp>(
        collapseShapeOp, collapseShapeOp.getSrcType(),
        collapseShapeOp.getSrc(),flattenReassociations);
      rewriter.setInsertionPoint(parallelInsertOp);
      rewriter.replaceOpWithNewOp<tensor::ParallelInsertSliceOp>(
          parallelInsertOp, newCollapseOp.getResult(),
          parallelInsertOp.getDest(), newOffsets, newSizes, newStrides);
    }
  }
  return;
}

/// Calculate the expanded shape of `dest` if it can be expanded with the inner
/// expanded sizes of `sliceStaticSizes`. Returns failure if such expansion is
/// not possible.
static LogicalResult getExpandedShape(SmallVector<ReassociationIndices> reInds,
                                      ArrayRef<int64_t> sliceStaticSizes,
                                      Value dest,
                                      SmallVector<int64_t> &expandedShape) {
  auto iter = expandedShape.begin();
  auto destType = cast<ShapedType>(dest.getType());

  for (auto [reassociations, destSize] :
       llvm::zip_equal(reInds, destType.getShape())) {
    int64_t totalInnerSize = 1;
    for (int i = reassociations.size() - 1; i > 0; --i) {
      int64_t expandedInnerSize = sliceStaticSizes[reassociations[i]];
      if (ShapedType::isDynamic(expandedInnerSize)) {
        return failure();
      }
      expandedShape.insert(iter, expandedInnerSize);
      totalInnerSize *= expandedInnerSize;
    }
    if (destSize % totalInnerSize != 0) {
      return failure();
    }
    expandedShape.insert(iter, destSize / totalInnerSize);
    iter = expandedShape.end();
  }
  return success();
}

struct ExpandDestinationForallOp final : OpRewritePattern<tensor::ParallelInsertSliceOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tensor::ParallelInsertSliceOp parallelInsertOp,
                                PatternRewriter &rewriter) const override {
    Location loc = parallelInsertOp.getLoc();
    MLIRContext *ctx = getContext();
    auto collapseOp =
        parallelInsertOp.getSource().getDefiningOp<tensor::CollapseShapeOp>();
    // No collapse op to hoist out.
    if (!collapseOp) {
      return failure();
    }

    // Ignore trivially foldable collapse ops.
    if (collapseOp.getSrcType().getRank()==collapseOp.getResultType().getRank()) {
      return failure();
    }

    // Get the destination to expand.
    Value insertDest = parallelInsertOp.getDest();

    // Verify that the users of destination are valid to expand and collect all such users.
    SmallVector<Operation *> expandableUsers;
    if (failed(verifyandCollectExpandableUsers(insertDest, collapseOp.getReassociationIndices(),expandableUsers))) {
      return failure();
    }

    SmallVector<ReassociationIndices> reInds =
        collapseOp.getReassociationIndices();
    SmallVector<int64_t> expandedShape;
    if (failed(getExpandedShape(collapseOp.getReassociationIndices(),
                                collapseOp.getSrcType().getShape(), insertDest,
                                expandedShape))) {
      return failure();
    }

    OpResult tiedResult = parallelInsertOp.getTiedOpResult();
    auto forallOp = dyn_cast<scf::ForallOp>(tiedResult.getOwner());
    if (!forallOp) {
      return failure();
    }
    SmallVector<Value> forallOutputs(forallOp.getOutputs());

    rewriter.setInsertionPoint(forallOp);

    if (forallOutputs.size() != 1) {
      return failure();
    }

    auto outOp = forallOutputs[0].getDefiningOp();
    if (!outOp) {
      return failure();
    }

    Type expandedType = RankedTensorType::get(
        expandedShape,
        cast<ShapedType>(outOp->getResult(0).getType()).getElementType());
    auto expanded = rewriter.create<tensor::ExpandShapeOp>(
        loc, expandedType, outOp->getResult(0),
        collapseOp.getReassociationIndices());

    // Arranged expanded dimS as a vector of each unexpanded dim so that they
    // can be used to delinearize them e.g for %expanded = tensor.expand_shape
    // %5 [[0, 1, 2], [3, 4]] output_shape [128, 8, 2, 640, 16] :
    // tensor<2048x10240xf32> into tensor<128x8x2x640x16xf32> We will construct
    // [[128, 8, 2],[640, 16]] This will be used to delinearize each unexpanded
    // Dim.
    SmallVector<SmallVector<OpFoldResult>> expandedDimsList;
    int expandDimBase = 0;
    for (auto reassociations : reInds) {
      SmallVector<OpFoldResult> expandedDims;
      for (int i = 0; i < reassociations.size(); i++) {
        expandedDims.push_back(
            getAsIndexOpFoldResult(ctx, expandedShape[expandDimBase + i]));
      }
      expandedDimsList.push_back(expandedDims);
      expandDimBase += reassociations.size();
    }

    scf::ForallOp newForallOp = rewriter.create<scf::ForallOp>(
        loc, forallOp.getMixedLowerBound(), forallOp.getMixedUpperBound(),
        forallOp.getMixedStep(), ValueRange{expanded},
        forallOp.getMappingAttr());

    auto newCollapseOp = rewriter.create<tensor::CollapseShapeOp>(
        loc, cast<ShapedType>(forallOp->getResult(0).getType()),
        newForallOp->getResult(0), reInds);
    rewriter.setInsertionPointToStart(forallOp.getBody());
    expandVerifiedExpandableUsers(ctx, rewriter, expandableUsers,
                                  expandedDimsList);
    SmallVector<Value> argReplacements(newForallOp.getInductionVars());
    for (auto forallIterArg : newForallOp.getRegionIterArgs()) {
      argReplacements.push_back(forallIterArg);
    }
    scf::InParallelOp parallelTerminator = newForallOp.getTerminator();
    parallelTerminator->erase();
    rewriter.mergeBlocks(forallOp.getBody(), newForallOp.getBody(),
                         argReplacements);
    forallOp->getResult(0).replaceAllUsesWith(newCollapseOp->getResult(0));
    return success();
  }
};

struct PropagateReshapesByExpansionPass final
    : impl::PropagateReshapesByExpansionPassBase<
          PropagateReshapesByExpansionPass> {
  void runOnOperation() override;
};
} // namespace

void PropagateReshapesByExpansionPass::runOnOperation() {
  MLIRContext *context = &getContext();

  {
    RewritePatternSet patterns(context);
    // Preemptively attempt to fold any reshapes into interface bindings if
    // possible to simplify subsequent reshape propagation.
    populateReshapeToInterfaceTensorPatterns(patterns);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }

  RewritePatternSet bubbleExpandShapePatterns(context);
  linalg::ControlFusionFn bubbleUpExpansionControlFn =
      [](OpOperand *fusedOperand) {
        Operation *producer = fusedOperand->get().getDefiningOp();
        Operation *consumer = fusedOperand->getOwner();

        // Block only if one of the operations has a lowering configuration
        // which means it likely expects tiling specific to its original shape.
        if (getLoweringConfig(producer) || getLoweringConfig(consumer)) {
          return false;
        }
        return true;
      };
  linalg::populateFoldReshapeOpsByExpansionPatterns(bubbleExpandShapePatterns,
                                                    bubbleUpExpansionControlFn);
  // Add patterns to do some additional cleanup (on top of canonicalizations
  // that can be done later) of reshape ops.
  tensor::populateFoldTensorEmptyPatterns(bubbleExpandShapePatterns);
  linalg::FillOp::getCanonicalizationPatterns(bubbleExpandShapePatterns,
                                              context);
  tensor::CollapseShapeOp::getCanonicalizationPatterns(
      bubbleExpandShapePatterns, context);
  tensor::EmptyOp::getCanonicalizationPatterns(bubbleExpandShapePatterns,
                                               context);
  tensor::ExpandShapeOp::getCanonicalizationPatterns(bubbleExpandShapePatterns,
                                                     context);
  populateReshapeToInterfaceTensorPatterns(bubbleExpandShapePatterns);
  bubbleExpandShapePatterns.add<ExpandDestinationForallOp>(context);
  if (failed(applyPatternsAndFoldGreedily(
          getOperation(), std::move(bubbleExpandShapePatterns)))) {
    getOperation()->emitOpError("Failed to propagate reshapes");
    return signalPassFailure();
  }
}

} // namespace mlir::iree_compiler

// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {


static SmallVector<NamedAttribute> PruneAttributeList(linalg::MatmulOp op) {
  auto opAttributes = op.getAttributeNames();
  llvm::StringSet<> elidedAttrs;
  elidedAttrs.insert(opAttributes.begin(), opAttributes.end());
  SmallVector<NamedAttribute> preservedAttrs;
  for (auto attr : op->getAttrs()) {
    if (elidedAttrs.count(attr.getName())) continue;
    preservedAttrs.push_back(attr);
  }
  return preservedAttrs;
}

namespace {

// Creates a linalg.generic that transposes input using permutation indices.
// Example: (M1, M0, N1, N0) -> (M1, N1, M0, N0) if indices = {0, 2, 1, 3}.
Value transpose(mlir::Location loc, PatternRewriter &rewriter,
                       Value input, ArrayRef<int64_t> indices) {
  RankedTensorType inputType = input.getType().cast<RankedTensorType>();
  auto nloops = indices.size();

  // TODO: use AffineMap::getPermutationMap?
  SmallVector<AffineExpr, 4> exprs = llvm::to_vector<4>(
      llvm::map_range(indices, [&](int64_t index) -> AffineExpr {
        return rewriter.getAffineDimExpr(index);
      }));

  ArrayRef<int64_t> inputShape = inputType.getShape();
  SmallVector<OpFoldResult, 4> targetShape;
  for (int i = 0; i < nloops; i++) {
    if (inputShape[indices[i]] == ShapedType::kDynamicSize) {
      targetShape.emplace_back(
          rewriter.create<tensor::DimOp>(loc, input, indices[i]));
    } else {
      targetShape.push_back(rewriter.getIndexAttr(inputShape[indices[i]]));
    }
  }

  Value outputTensor = rewriter.create<linalg::InitTensorOp>(
      loc, targetShape, inputType.getElementType());

  SmallVector<StringRef, 4> loopAttributeTypes(nloops, "parallel");

  SmallVector<AffineMap, 2> indexingMaps = {
      inversePermutation(
          AffineMap::get(nloops, 0, exprs, rewriter.getContext())),
      AffineMap::getMultiDimIdentityMap(nloops, rewriter.getContext())};

  auto transposedOp = rewriter.create<linalg::GenericOp>(
      loc, outputTensor.getType(),
      /*inputs=*/input, /*outputs=*/outputTensor, indexingMaps,
      loopAttributeTypes,
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
        nestedBuilder.create<linalg::YieldOp>(nestedLoc, args[0]);
      });

  return transposedOp.getResult(0);
};

// Creates a linalg.generic that does the compute after packedB layout
Value computeGeneric(mlir::Location loc, PatternRewriter &rewriter,
                       Value &inputLhs, Value &inputRhs, linalg::MatmulOp &matmulOp) {
  auto nloops = 4;
  std::array<int64_t,3> indicesLhs = {0,2,3};
  std::array<int64_t,3> indicesRhs = {2,1,3};
  std::array<int64_t,2> indicesResult = {0,1};
  SmallVector<AffineExpr, 4> exprsLhs = llvm::to_vector<4>(
      llvm::map_range(indicesLhs, [&](int64_t index) -> AffineExpr {
        return rewriter.getAffineDimExpr(index);
      }));
  SmallVector<AffineExpr, 4> exprsRhs = llvm::to_vector<4>(
      llvm::map_range(indicesRhs, [&](int64_t index) -> AffineExpr {
        return rewriter.getAffineDimExpr(index);
      }));
  SmallVector<AffineExpr, 4> exprsResult = llvm::to_vector<4>(
      llvm::map_range(indicesResult, [&](int64_t index) -> AffineExpr {
        return rewriter.getAffineDimExpr(index);
      }));
  std::array<AffineMap,3> indexingMaps={
    AffineMap::get(nloops, 0, exprsLhs, rewriter.getContext()),
    AffineMap::get(nloops, 0, exprsRhs, rewriter.getContext()),
    AffineMap::get(nloops, 0, exprsResult, rewriter.getContext()),
  };
  SmallVector<StringRef> iterators{"parallel", "parallel","reduction","reduction"};
  //Block &payload = matmulOp.region().front();
 SmallVector<Value> newOperands={inputLhs,inputRhs};
auto newOp = rewriter.create<linalg::GenericOp>(
        matmulOp.getLoc(), matmulOp->getResultTypes(), newOperands,matmulOp.getOutputOperand(0)->get(),
        indexingMaps, iterators,/*bodyBuild=*/nullptr,PruneAttributeList(matmulOp));
/*ArrayRef<StringRef> odsAttrs = matmulOp.getAttributeNames();
for (NamedAttribute kv : matmulOp->getAttrs()) {
  if (!llvm::is_contained(odsAttrs, kv.getName().getValue())) {
        newOp->setAttr(kv.getName(), kv.getValue());
  }
}*/       
rewriter.inlineRegionBefore(matmulOp.region(), newOp.region(),
                                newOp.region().begin());


return newOp.getResult(0);
};

struct PackedBLinalgMatmulOp : public OpRewritePattern<linalg::MatmulOp> {
  using OpRewritePattern<linalg::MatmulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::MatmulOp matmulOp,
                                PatternRewriter &rewriter) const override {
    auto loc = matmulOp.getLoc();
    auto lhs = matmulOp.getInputOperand(0)->get();
    lhs.dump();
    auto rhs = matmulOp.getInputs()[1];
    auto rhsType = rhs.getType().cast<RankedTensorType>();
    auto rhsShape = rhsType.getShape();
    auto lhsType = lhs.getType().cast<RankedTensorType>();
    auto lhsShape = lhsType.getShape();
    //SmallVector<AffineMap> newIndexingMaps;

    RankedTensorType lhsReshapeType = RankedTensorType::get({lhsShape[0], lhsShape[1]/2,2}, lhsType.getElementType());
    //auto newRhs = rhs.reshape()
    std::array<ReassociationIndices, 2> lhsExpandIndices = {
      ReassociationIndices{0}, ReassociationIndices{1,2}};
  Value newLhs = rewriter.create<tensor::ExpandShapeOp>(
      loc, lhsReshapeType, lhs, lhsExpandIndices);

    RankedTensorType rhsReshapeType = RankedTensorType::get({rhsShape[0]/2, 2, rhsShape[1]}, rhsType.getElementType());
    //auto newRhs = rhs.reshape()
    std::array<ReassociationIndices, 2> rhsExpandIndices = {
      ReassociationIndices{0,1}, ReassociationIndices{2}};
  Value newRhs = rewriter.create<tensor::ExpandShapeOp>(
      loc, rhsReshapeType, rhs, rhsExpandIndices);
    //newLhs.dump();
    //newRhs.dump();
  Value newTransposeRhs = transpose(loc,rewriter,newRhs, {0,2,1});
   //newTransposeRhs.dump();
  Value result = computeGeneric(loc,rewriter,newLhs,newTransposeRhs,matmulOp);
  rewriter.replaceOp(matmulOp, ArrayRef<Value>{result});
   //result.dump();
    return success();
  }
};

struct PackedBLayoutPass
    : public PackedBLayoutPassBase<PackedBLayoutPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(&getContext());
    patterns.insert<PackedBLinalgMatmulOp>(context);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createPackedBLayoutPass() {
  return std::make_unique<PackedBLayoutPass>();
}

}  // namespace iree_compiler
}  // namespace mlir

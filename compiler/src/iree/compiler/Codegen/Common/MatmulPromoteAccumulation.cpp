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
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
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

struct PromoteAccMatmulOp : public OpRewritePattern<linalg::MatmulOp> {
  using OpRewritePattern<linalg::MatmulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::MatmulOp matmulOp,
                                PatternRewriter &rewriter) const override {
    auto loc = matmulOp.getLoc();
    //auto lhs = matmulOp.getInputOperand(0)->get();
    //lhs.dump();
    //auto rhs = matmulOp.getInputs()[1];
    //auto rhsType = rhs.getType().cast<RankedTensorType>();
    //auto rhsShape = rhsType.getShape();
    //auto lhsType = lhs.getType().cast<RankedTensorType>();
    //auto lhsShape = lhsType.getShape();
    auto acc = matmulOp.getOutputOperand(0)->get();
    auto accType = acc.getType().cast<RankedTensorType>();
    if(accType.getElementType().getIntOrFloatBitWidth()==32){
      return failure();
    }
    auto accTypefp32 = RankedTensorType::get(accType.getShape(), Float32Type::get(accType.getElementType().getContext()));
    accTypefp32.dump();
    std::array<int64_t,3> indices = {0,1};
    SmallVector<AffineExpr, 4> exprs = llvm::to_vector<4>(
      llvm::map_range(indices, [&](int64_t index) -> AffineExpr {
        return rewriter.getAffineDimExpr(index);
      }));

  Value outputTensor = rewriter.create<linalg::InitTensorOp>(
      loc, accType.getShape(), Float32Type::get(accType.getElementType().getContext()));

  SmallVector<StringRef, 4> loopAttributeTypes(2, "parallel");

  SmallVector<AffineMap, 2> indexingMaps = {
      AffineMap::getMultiDimIdentityMap(2, rewriter.getContext()),
      AffineMap::getMultiDimIdentityMap(2, rewriter.getContext())};

  auto genericf32Op = rewriter.create<linalg::GenericOp>(
      loc, outputTensor.getType(),
      /*inputs=*/acc, /*outputs=*/outputTensor, indexingMaps,
      loopAttributeTypes,
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
        Value castVal = nestedBuilder.create<arith::ExtFOp>(nestedLoc, 
          Float32Type::get(accType.getElementType().getContext()), 
          args[0]);
        nestedBuilder.create<linalg::YieldOp>(nestedLoc, castVal);
      });
  
  auto newMatmulOp = rewriter.create<linalg::MatmulOp>(
    loc,accTypefp32,matmulOp.getInputs(), ArrayRef<Value>{genericf32Op.getResult(0)},
    PruneAttributeList(matmulOp)
  );

  Value outputTensorf16 = rewriter.create<linalg::InitTensorOp>(
      loc, accType.getShape(), accType.getElementType());

  
  auto genericf16Op = rewriter.create<linalg::GenericOp>(
      loc, outputTensorf16.getType(),
      /*inputs=*/newMatmulOp.getResult(0), /*outputs=*/outputTensorf16, indexingMaps,
      loopAttributeTypes,
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
        Value castVal = nestedBuilder.create<arith::TruncFOp>(nestedLoc, 
          accType.getElementType(), args[0]);
        nestedBuilder.create<linalg::YieldOp>(nestedLoc, castVal);
      });

  rewriter.replaceOp(matmulOp, ArrayRef<Value>{genericf16Op.getResult(0)});

    return success();
  }
};

struct MatmulPromoteAccumulationPass
    : public MatmulPromoteAccumulationPassBase<MatmulPromoteAccumulationPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithmeticDialect,linalg::LinalgDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(&getContext());
    patterns.insert<PromoteAccMatmulOp>(context);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createMatmulPromoteAccumulationPass() {
  return std::make_unique<MatmulPromoteAccumulationPass>();
}

}  // namespace iree_compiler
}  // namespace mlir

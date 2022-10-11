// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "iree/compiler/Codegen/SPIRV/Utils.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVAttributes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVTypes.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"
#include <iostream>

namespace mlir {
namespace iree_compiler {

namespace {

class FoldExtIntoContract
    : public OpRewritePattern<spirv::INTELJointMatrixLoadOp> {
 public:
  using OpRewritePattern<spirv::INTELJointMatrixLoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(spirv::INTELJointMatrixLoadOp op,
                                PatternRewriter& rewriter) const override {
    std::cout<<"In the SPIRV level pass for SPIRV\n";
    auto opLayout= op.getLayout();
    auto result = op.getResult();
    auto jointMatrixType =
          result.getType().dyn_cast<spirv::JointMatrixINTELType>();
    
    auto resultLayout = jointMatrixType.getMatrixLayout();

    if(resultLayout==opLayout){
      return failure();
    }
    // use former if not packing, later if packing
     /* auto newJointMatrixType = spirv::JointMatrixINTELType::get(
                jointMatrixType.getElementType(), spirv::Scope::Subgroup, jointMatrixType.getRows(),
                jointMatrixType.getColumns(), opLayout);*/
      auto newJointMatrixType = spirv::JointMatrixINTELType::get(
                jointMatrixType.getElementType(), spirv::Scope::Subgroup, jointMatrixType.getColumns(),
                jointMatrixType.getRows(), opLayout);
      rewriter.replaceOpWithNewOp<spirv::INTELJointMatrixLoadOp> (op,newJointMatrixType,
      op.getPointer(),op.getStride(), opLayout, spirv::Scope::Subgroup,
                spirv::MemoryAccessAttr(),IntegerAttr());
      return success();

  }
};

class CastJointMatrixLoadProducerPtr
    : public OpRewritePattern<spirv::INTELJointMatrixLoadOp> {
 public:
  using OpRewritePattern<spirv::INTELJointMatrixLoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(spirv::INTELJointMatrixLoadOp op,
                                PatternRewriter& rewriter) const override {
    
     auto opPointer= op.getPointer();
     auto pointerType = opPointer.getType().cast<spirv::PointerType>();
     auto pointerStorage = pointerType.getStorageClass();
     if (pointerStorage == spirv::StorageClass::Generic)
      return failure();
    Location loc = op.getLoc();
    auto newPointerType = spirv::PointerType::get(pointerType.getPointeeType(),spirv::StorageClass::Generic);
    Value newPtr = rewriter.create<spirv::PtrCastToGenericOp>(loc,newPointerType,opPointer);
      rewriter.replaceOpWithNewOp<spirv::INTELJointMatrixLoadOp> (op,op.getResult().getType(),
      newPtr,op.getStride(), op.getLayout(), spirv::Scope::Subgroup,
                spirv::MemoryAccessAttr(),IntegerAttr());
    return success();
  }
};

class CastJointMatrixStoreProducerPtr
    : public OpRewritePattern<spirv::INTELJointMatrixStoreOp> {
 public:
  using OpRewritePattern<spirv::INTELJointMatrixStoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(spirv::INTELJointMatrixStoreOp op,
                                PatternRewriter& rewriter) const override {
    
     auto opPointer= op.getPointer();
     auto pointerType = opPointer.getType().cast<spirv::PointerType>();
     auto pointerStorage = pointerType.getStorageClass();
     if (pointerStorage == spirv::StorageClass::Generic)
      return failure();
    Location loc = op.getLoc();
    auto newPointerType = spirv::PointerType::get(pointerType.getPointeeType(),spirv::StorageClass::Generic);
    Value newPtr = rewriter.create<spirv::PtrCastToGenericOp>(loc,newPointerType,opPointer);
      rewriter.replaceOpWithNewOp<spirv::INTELJointMatrixStoreOp> (op,
      newPtr,op.getObject(),op.getStride(), op.getLayout(), spirv::Scope::Subgroup,
                spirv::MemoryAccessAttr(),IntegerAttr());
    return success();
  }
};

struct SPIRVMatchJointLoadPass
    : public SPIRVMatchJointLoadBase<SPIRVMatchJointLoadPass> {
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();

    {
      RewritePatternSet patterns(&getContext());
      patterns.add<FoldExtIntoContract>(&getContext());
      patterns.add<CastJointMatrixLoadProducerPtr>(&getContext());
      patterns.add<CastJointMatrixStoreProducerPtr>(&getContext());
      if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
        return signalPassFailure();
      }

    }

  }


};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createSPIRVMatchJointLoadPass() {
  return std::make_unique<SPIRVMatchJointLoadPass>();
}

}  // namespace iree_compiler
}  // namespace mlir

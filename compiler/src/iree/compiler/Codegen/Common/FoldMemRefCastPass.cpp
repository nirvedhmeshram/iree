// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


#include <memory>

#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h"
#include "iree/compiler/Codegen/SPIRV/Utils.h"

#define DEBUG_TYPE "iree-fold-MemRef-cast"

namespace mlir {
namespace iree_compiler {

namespace {
/// Pattern to lower operations that become a no-ops at this level.
template <typename OpTy>
struct FoldAsNoOp final : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;
  LogicalResult matchAndRewrite(
      OpTy op, PatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, op.getSource());
    return success();
  }
};
//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct FoldMemRefCastPass
    : public FoldMemRefCastBase<FoldMemRefCastPass> {
  FoldMemRefCastPass() {}
  FoldMemRefCastPass(const FoldMemRefCastPass &pass) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override {
    // First flatten the dimensions of subspan op and their consumer load/store
    // ops. This requires setting up conversion targets with type converter.
  //ModuleOp moduleOp = getOperation();
  MLIRContext &context = getContext();
  /*spirv::TargetEnvAttr targetAttr = getSPIRVTargetEnvAttr(moduleOp);
  moduleOp->setAttr(spirv::getTargetEnvAttrName(), targetAttr);

  SPIRVConversionOptions options = {};
  options.enableFastMathMode = false;
  options.use64bitIndex = true;
  SPIRVTypeConverter typeConverter(targetAttr, options);*/

    
  RewritePatternSet patterns(&context);
  patterns.add<FoldAsNoOp<memref::CastOp>>(&context);
    

    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace


std::unique_ptr<OperationPass<ModuleOp>> createFoldMemRefCastPass() {
  return std::make_unique<FoldMemRefCastPass>();
}

}  // namespace iree_compiler
}  // namespace mlir

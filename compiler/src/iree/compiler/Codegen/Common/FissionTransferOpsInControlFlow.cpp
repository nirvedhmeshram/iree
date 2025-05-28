// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Common/Transforms.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-fission-transfer-ops-in-control-flow"

#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE << "]: ")

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_FISSIONTRANSFEROPSINCONTROLFLOWPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

void replaceIterationVariable(Operation *op, Value iterArg, Value constant) {
  for (auto &operand : op->getOpOperands()) {
    if (operand.get() == iterArg) {
      operand.set(constant);
    }
  }
}

SetVector<Operation *> collectBackwardSliceInControlFlow(
    Operation *op, Operation *parentOp) {
  BackwardSliceOptions options;
  options.inclusive = true;
  options.filter = [&](Operation *op) {
    return parentOp == op->getParentOp();
  };
  SetVector<Operation *> slice;
  getBackwardSlice(op, &slice, options);
  return slice;
}

void splitTransferOpsFromControlFlow(PatternRewriter &rewriter,
    vector::TransferReadOp readOp, vector::TransferWriteOp writeOp, scf::ForOp forOp) {
  rewriter.setInsertionPoint(forOp);
  auto alloca = rewriter.create<memref::AllocaOp>(  
              //readOp.getLoc(), writeOp.getBase().getType()
              readOp.getLoc(), cast<MemRefType>(writeOp.getBase().getType())
      );
  SetVector<Operation *> readSlice =
      collectBackwardSliceInControlFlow(readOp, forOp);
  auto readLoop = rewriter.create<scf::ForOp>(
      readOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
      forOp.getStep(), forOp.getRegionIterArgs());
  //std::reverse(readSlice.begin(), readSlice.end());
  //reverse the readSLice
  SmallVector<Operation *> reversedReadSlice(readSlice.rbegin(),
                                                  readSlice.rend());
  // Move the readSlice into the readLoop.
  for (Operation *op : reversedReadSlice) {
      op->moveBefore(readLoop.getBody(), readLoop.getBody()->begin());
  }
  rewriter.setInsertionPointToStart(readLoop.getBody());
  auto newTransferWrite = writeOp.clone();      
  newTransferWrite->setOperand(1, alloca);
  //auto endOp = 
  //auto writeLoop = forOp.clone();
  rewriter.setInsertionPoint(forOp);
  for (Operation &op : forOp.getBody()->getOperations()) {
    replaceIterationVariable(&op, readOp->getResult(0),
                            alloca);
  }
  DBGS() << "Read loop: " << "\n";
  readLoop.dump();
  DBGS() << "Write loop: " << "\n";
  forOp.dump();

  //forOp.erase();
}

void hoistTransferReadAndDependencies(scf::ForOp forOp) {
  OpBuilder builder(forOp.getContext());
  //std::vector<Operation *> dependencies;
  DenseSet<Operation *> visited;
  Value iterArg = forOp.getInductionVar();

  //SetVector<Operation *> slice;
  SetVector<Operation *> dependencies;
  BackwardSliceOptions options;
  options.inclusive = true;
  options.filter = [&](Operation *op) {
    return forOp == op->getParentOp();
  };
  // Find transfer_read operations and relevant dependencies within the loop.
  forOp.walk([&](vector::TransferReadOp readOp) {
    getBackwardSlice(readOp.getOperation(), &dependencies, options);
  });

  builder.setInsertionPoint(forOp);
  Value zero = builder.create<arith::ConstantIndexOp>(forOp.getLoc(), 0);

  // Hoist dependencies in the order they were collected.
  for (Operation *dep : dependencies) {
    replaceIterationVariable(dep, iterArg, zero);
    dep->moveBefore(forOp);
  }
}

struct FissionTarget{
  Operation *parent;
  vector::TransferReadOp readOp;
  vector::TransferWriteOp writeOp;
};

static FailureOr<SmallVector<FissionTarget>> populateFissionTargets(
    vector::TransferReadOp readOp) {
  auto parentOp = readOp->getParentOp();
  if (!parentOp || !isa<scf::ForOp, scf::IfOp, scf::WhileOp>(parentOp)) {
    return failure();
  }

  auto base = readOp.getBase();
  auto addrspace = cast<MemRefType>(base.getType()).getMemorySpace();
  if (gpu::GPUDialect::isWorkgroupMemoryAddressSpace(addrspace) ||
      gpu::GPUDialect::isPrivateAddressSpace(addrspace)) {
    return failure();
  }

  ForwardSliceOptions options;
  options.inclusive = false;
  options.filter = [&](Operation *op) {
    return parentOp == op->getParentOp();
  };
  SetVector<Operation *> slice;
  getForwardSlice(readOp.getOperation(), &slice, options);

  SmallVector<FissionTarget> fissionTargets;
  for (Operation *op : slice) {
    if (auto writeOp = dyn_cast<vector::TransferWriteOp>(op)) {
      auto writeBase = writeOp.getBase();
      auto writeAddrspace =
          cast<MemRefType>(writeBase.getType()).getMemorySpace();
      if (gpu::GPUDialect::isPrivateAddressSpace(writeAddrspace)) {
        // Only consider transfer_write ops that are in the same address space.
        fissionTargets.push_back(
            {parentOp, readOp, writeOp});
      DBGS() << "Fissioning target readOp: " << readOp
             << " and writeOp: " << writeOp
             //<< " in parent operation: " << cast<scf::ForOp>(target.parent)
             << "\n";

      }
    }
  }
  return fissionTargets;
}


struct FissionTransferOpsInControlFlowPattern final
    : public OpRewritePattern<vector::TransferReadOp> {
  using OpRewritePattern<vector::TransferReadOp>::OpRewritePattern;


  LogicalResult matchAndRewrite(vector::TransferReadOp readOp,
                                PatternRewriter &rewriter) const override {
    auto FissionTargets = populateFissionTargets(readOp);
    if (failed(FissionTargets) || FissionTargets->empty()) {
      return failure();
    }

    for (const FissionTarget &target : *FissionTargets) {
      if (isa<scf::ForOp>(target.parent)) {
        // Hoist the transfer read and dependencies in the for loop.
        //hoistTransferReadAndDependencies(cast<scf::ForOp>(target.parent));
        splitTransferOpsFromControlFlow(rewriter, readOp, target.writeOp,
                                        cast<scf::ForOp>(target.parent));
      } else {
        return rewriter.notifyMatchFailure(
            target.parent, "Unsupported parent operation for fission");
      }
    }

    return success();
  }
};

struct FissionTransferOpsInControlFlowPass final
    : impl::FissionTransferOpsInControlFlowPassBase<
          FissionTransferOpsInControlFlowPass> {
public:
  using impl::FissionTransferOpsInControlFlowPassBase<
      FissionTransferOpsInControlFlowPass>::FissionTransferOpsInControlFlowPassBase;
  void runOnOperation() override;
};

} // namespace
void populateFissionTransferOpsInControlFlowPatterns(
    RewritePatternSet &patterns) {
  patterns.add<
      FissionTransferOpsInControlFlowPattern>(
      patterns.getContext());
}

void FissionTransferOpsInControlFlowPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  populateFissionTransferOpsInControlFlowPatterns(
      patterns);
  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
    return signalPassFailure();
  }
}
  
} // namespace mlir::iree_compiler

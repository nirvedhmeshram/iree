// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Codegen/LLVMGPU/KernelConfig.h"
#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_LLVMGPUSPECIALIZEUNALIGNEDMATMULPASS
#include "iree/compiler/Codegen/LLVMGPU/Passes.h.inc"

namespace {
/// Selects a lowering strategy for taking a hal.executable.variant operation
/// to scalar/native-vector code.
class LLVMGPUSpecializeUnalignedMatmulPass final
    : public impl::LLVMGPUSpecializeUnalignedMatmulPassBase<
          LLVMGPUSpecializeUnalignedMatmulPass> {
public:
  using impl::LLVMGPUSpecializeUnalignedMatmulPassBase<
      LLVMGPUSpecializeUnalignedMatmulPass>::LLVMGPUSpecializeUnalignedMatmulPassBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<IREE::Codegen::IREECodegenDialect, IREE::GPU::IREEGPUDialect>();
  }

  void runOnOperation() override;
};
} // namespace


void LLVMGPUSpecializeUnalignedMatmulPass::runOnOperation() {
  //BackwardSliceOptions options;
  SmallVector<IREE::Flow::DispatchTensorStoreOp> storeOps;
  auto moduleOp = getOperation();
  for (auto funcOp : moduleOp.getOps<FunctionOpInterface>()) {
    funcOp.walk([&](IREE::Flow::DispatchTensorStoreOp storeOp) {
      storeOps.push_back(storeOp);
    });
  }
  storeOps[0]->dump();

}

} // namespace mlir::iree_compiler

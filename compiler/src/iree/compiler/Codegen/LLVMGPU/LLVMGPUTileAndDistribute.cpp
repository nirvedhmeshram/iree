// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree-dialects/Dialect/LinalgExt/Passes/Transforms.h"
#include "iree-dialects/Dialect/LinalgExt/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Dialect/LoweringConfig.h"
#include "iree/compiler/Codegen/LLVMGPU/KernelConfig.h"
#include "iree/compiler/Codegen/LLVMGPU/TilingUtils.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Codegen/Utils/MarkerUtils.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Support/MathExtras.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/SideEffectUtils.h"

using mlir::iree_compiler::IREE::LinalgExt::TilingPatterns;

#define DEBUG_TYPE "iree-llvmgpu-tile-and-distribute"

  static llvm::cl::opt<bool> clCodegenDisablePromotion(
      "iree-codegen-disable-promotion", llvm::cl::desc("not use shared memory"),
      llvm::cl::init(false));

namespace mlir {
namespace iree_compiler {

/// Patterns for workgroup level tiling. Workgroup tiling is done at the flow
/// level but we may have extra tiling for the reduction dimension. Therefore we
/// tile again without distributing.
static void populateTilingReductionPatterns(RewritePatternSet &patterns) {
  auto tileSizesFn = [&](OpBuilder &builder,
                         Operation *op) -> SmallVector<Value, 4> {
    auto interfaceOp = cast<PartitionableLoopsInterface>(*op);
    auto partitionedLoops =
        interfaceOp.getPartitionableLoops(kNumMaxParallelDims);
    SmallVector<Value, 4> tileSizes = getTileSizes(builder, op, 0);
    auto zero = builder.create<arith::ConstantIndexOp>(op->getLoc(), 0);
    for (unsigned depth : partitionedLoops) {
      if (depth < tileSizes.size()) {
        tileSizes[depth] = zero;
      }
    }
    return tileSizes;
  };

  auto tilingOptions = linalg::LinalgTilingOptions()
                           .setLoopType(linalg::LinalgTilingLoopType::Loops)
                           .setTileSizeComputationFunction(tileSizesFn);
  MLIRContext *context = patterns.getContext();

  linalg::LinalgTransformationFilter filter(
      ArrayRef<StringAttr>{
          StringAttr::get(context, getWorkgroupMemoryMarker())},
      StringAttr::get(context, getWorkgroupKTiledMarker()));
  filter.setMatchByDefault();
  TilingPatterns<linalg::MatmulOp, linalg::BatchMatmulOp,
                 linalg::GenericOp>::insert(patterns, tilingOptions, filter);
}

/// Return the tile size associated to one thread or warp based on the number of
/// element in the group.
static SmallVector<Value, 4> calculateDistributedTileSize(
    ArrayRef<int64_t> numElements, OpBuilder &builder, Operation *operation) {
  SmallVector<int64_t> blockTileSize = getTileSizes(operation, 0);
  SmallVector<Value, 4> tileSizesVal;
  // Use partitionedLoop to know what loop needs to be distributed.
  auto interfaceOp = cast<PartitionableLoopsInterface>(operation);
  auto partitionedLoops =
      interfaceOp.getPartitionableLoops(kNumMaxParallelDims);
  if (partitionedLoops.empty()) {
    return tileSizesVal;
  }
  auto zero = builder.create<arith::ConstantIndexOp>(operation->getLoc(), 0);
  tileSizesVal.resize(
      cast<TilingInterface>(operation).getLoopIteratorTypes().size(), zero);

  // partitionedLoops contains the dimensions we want to distribute.
  // We are distributing them in order onto the different workgroup
  // dimensions.
  SmallVector<int64_t> distributedDim(numElements.begin(), numElements.end());
  distributedDim.resize(partitionedLoops.size());
  unsigned idIdx = 0;
  std::reverse(distributedDim.begin(), distributedDim.end());
  for (unsigned depth : partitionedLoops) {
    if (depth >= blockTileSize.size()) continue;
    tileSizesVal[depth] = builder.create<arith::ConstantIndexOp>(
        operation->getLoc(),
        llvm::divideCeil(blockTileSize[depth], distributedDim[idIdx++]));
    if (idIdx == kNumMaxParallelDims) break;
  }
  return tileSizesVal;
}

/// Patterns for warp level tiling.
static void populateTilingToWarpPatterns(
    RewritePatternSet &patterns, SmallVectorImpl<int64_t> &workgroupSize) {
  std::array<int64_t, 3> warpPerWorkgroup = {
      workgroupSize[0] / kWarpSize, workgroupSize[1], workgroupSize[2]};

  linalg::TileSizeComputationFunction getInnerTileSizeFn =
      [warpPerWorkgroup](OpBuilder &builder, Operation *operation) {
        return calculateDistributedTileSize(warpPerWorkgroup, builder,
                                            operation);
      };
  auto getWarpProcInfoFn = [warpPerWorkgroup](
                               OpBuilder &builder, Location loc,
                               ArrayRef<Range> parallelLoopRanges) {
    return getSubgroupIdsAndCounts(builder, loc, parallelLoopRanges.size(),
                                   warpPerWorkgroup);
  };
  linalg::LinalgLoopDistributionOptions warpDistributionOptions;
  warpDistributionOptions.procInfo = getWarpProcInfoFn;

  auto tilingOptions = linalg::LinalgTilingOptions()
                           .setLoopType(linalg::LinalgTilingLoopType::Loops)
                           .setTileSizeComputationFunction(getInnerTileSizeFn)
                           .setDistributionOptions(warpDistributionOptions);
  MLIRContext *context = patterns.getContext();
  linalg::LinalgTransformationFilter filter(
      {StringAttr::get(context, getWorkgroupKTiledMarker()),
       StringAttr::get(context, getWorkgroupMemoryMarker())},
      StringAttr::get(context, getVectorizeMarker()));
  filter.setMatchByDefault();
  TilingPatterns<linalg::MatmulOp, linalg::FillOp, linalg::BatchMatmulOp,
                 linalg::GenericOp>::insert(patterns, tilingOptions, filter);
}

/// Patterns for thread level tiling.
static void populateTilingToInvocationPatterns(
    RewritePatternSet &patterns, SmallVectorImpl<int64_t> &workgroupSize) {
  linalg::TileSizeComputationFunction getInnerTileSizeFn =
      [&](OpBuilder &builder, Operation *operation) {
        return calculateDistributedTileSize(workgroupSize, builder, operation);
      };
  auto getThreadProcInfoFn = [&workgroupSize](
                                 OpBuilder &builder, Location loc,
                                 ArrayRef<Range> parallelLoopRanges) {
    return getGPUThreadIdsAndCounts(builder, loc, parallelLoopRanges.size(),
                                    workgroupSize);
  };
  linalg::LinalgLoopDistributionOptions invocationDistributionOptions;
  invocationDistributionOptions.procInfo = getThreadProcInfoFn;

  auto tilingOptions =
      linalg::LinalgTilingOptions()
          .setLoopType(linalg::LinalgTilingLoopType::Loops)
          .setTileSizeComputationFunction(getInnerTileSizeFn)
          .setDistributionOptions(invocationDistributionOptions);

  MLIRContext *context = patterns.getContext();
  linalg::LinalgTransformationFilter f(
      {StringAttr::get(context, getWorkgroupKTiledMarker()),
       StringAttr::get(context, getWorkgroupMemoryMarker())},
      StringAttr::get(context, getVectorizeMarker()));
  f.addFilter([](Operation *op) {
     // FFT doesn't support second level of tiling yet.
     return success(!isa<IREE::LinalgExt::FftOp>(op));
   }).setMatchByDefault();
  patterns.insert<linalg::LinalgTilingPattern,
                  IREE::LinalgExt::TilingInterfaceTilingPattern>(
      context, tilingOptions, f);
}

static LogicalResult copyToWorkgroupMemory(OpBuilder &b, Value src, Value dst) {
  Operation *copyOp = b.create<memref::CopyOp>(src.getLoc(), src, dst);
  setMarker(copyOp, getCopyToWorkgroupMemoryMarker());
  return success();
}

using PromotionFilterFunction = std::function<LogicalResult(Operation *op)>;

/// Returns true if op is appropriate contract for promotion.
static LogicalResult contractOpFilter(Operation *op) {
  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  if (!linalgOp) return failure();
  // Limit promotion to matmul and batch matmul, there may be generic
  // ops with more batch dimensions we didn't distribute and therefore
  // cannot find a higher bound.
  return success(linalg::isaContractionOpInterface(op) &&
                 linalgOp.getNumParallelLoops() >= 2 &&
                 linalgOp.getNumParallelLoops() <= 3);
}

template <typename T>
using LinalgPromotionPattern =
    mlir::iree_compiler::IREE::LinalgExt::LinalgPromotionPattern<T>;
static void populatePromotionPatterns(MLIRContext *context,
                                      RewritePatternSet &patterns,
                                      PromotionFilterFunction filterFunction,
                                      ArrayRef<int64_t> operandsToPromote) {
  patterns.insert<LinalgPromotionPattern<linalg::MatmulOp>,
                  LinalgPromotionPattern<linalg::BatchMatmulOp>,
                  LinalgPromotionPattern<linalg::GenericOp>>(
      context,
      linalg::LinalgPromotionOptions()
          .setAllocationDeallocationFns(allocateWorkgroupMemory,
                                        deallocateWorkgroupMemory)
          .setCopyInOutFns(copyToWorkgroupMemory, copyToWorkgroupMemory)
          .setOperandsToPromote(operandsToPromote)
          .setUseFullTileBuffers({false, false}),
      linalg::LinalgTransformationFilter(
          {StringAttr::get(context, getWorkgroupKTiledMarker())},
          StringAttr::get(context, getWorkgroupMemoryMarker()))
          .setMatchByDefault()
          .addFilter(filterFunction));
}

static bool propagateCopyDestIntoProducerFill(memref::CopyOp copyOp) {
  // Look for a fill Op writing into the copyOp source.
  Operation *prevOp = copyOp->getPrevNode();
  while (prevOp) {
    if (isSideEffectFree(prevOp)) {
      prevOp = prevOp->getPrevNode();
      continue;
    }

    auto fillOp = dyn_cast<linalg::FillOp>(prevOp);
    if (!fillOp) break;
    if (fillOp.output() != copyOp.getSource()) break;
    // Move the fillOp and change the destination to the copy destination.
    fillOp->moveBefore(copyOp);
    fillOp.getOutputsMutable().assign(copyOp.getTarget());
    return true;
  }
  return false;
}

// Split input/output operand from copy from shared memory into a separate
// input.
static void insertInputValueIntoGeneric(Value source, linalg::GenericOp op) {
  SmallVector<Value> newOperands;
  SmallVector<AffineMap> maps;
  for (OpOperand *in : op.getInputOperands()) {
    newOperands.push_back(in->get());
    maps.push_back(op.getTiedIndexingMap(in));
  }
  newOperands.push_back(source);
  assert(op.getNumOutputs() == 1);
  OpOperand *outOperand = op.getOutputOperand(0);
  maps.push_back(op.getTiedIndexingMap(outOperand));
  maps.push_back(op.getTiedIndexingMap(outOperand));
  Location loc = op.getLoc();
  SmallVector<StringRef> iterTypes(op.getNumLoops(),
                                   getParallelIteratorTypeName());
  OpBuilder builder(op);
  auto newOp = builder.create<linalg::GenericOp>(
      loc, newOperands, outOperand->get(), maps, iterTypes);
  newOp.getRegion().getBlocks().splice(newOp.getRegion().begin(),
                                       op.getRegion().getBlocks());

  Block &payload = newOp.getRegion().front();
  payload.addArgument(payload.getArguments().back().getType(), loc);
  setMarker(newOp, getCopyToWorkgroupMemoryMarker());
}

/// Propagate the shared memory copy into the consumer op if it's a fully
/// parallel linalg.generic.
static bool propagateCopySourceIntoConsumerGeneric(
    memref::CopyOp copyOp, SmallVector<Operation *> &toDelete) {
  // Look for a generic Op reading the copyOp target.
  Operation *nextOp = copyOp->getNextNode();
  while (nextOp) {
    if (isSideEffectFree(nextOp)) {
      nextOp = nextOp->getNextNode();
      continue;
    }
    auto consumer = dyn_cast<linalg::GenericOp>(nextOp);
    if (!consumer || consumer.getNumOutputs() != 1 ||
        !consumer.getTiedIndexingMap(consumer.getOutputOperand(0)).isIdentity())
      break;
    if (*consumer.outputs().begin() != copyOp.getTarget()) break;
    insertInputValueIntoGeneric(copyOp.getSource(), consumer);
    toDelete.push_back(consumer);
    return true;
  }
  return false;
}

/// Transformation to propagate FillOp + CopyOp to temp allocation.
/// This is needed because we are doing promotion to shared memory on buffers.
/// This is a fragile and temporary solution until we move to be able to do this
/// kind of transformations on tensors.
static void propagateSharedMemCopy(func::FuncOp funcOp) {
  SmallVector<Operation *> toDelete;
  funcOp.walk([&toDelete](memref::CopyOp copyOp) {
    if (hasMarker(copyOp, getCopyToWorkgroupMemoryMarker())) {
      if (propagateCopyDestIntoProducerFill(copyOp) ||
          propagateCopySourceIntoConsumerGeneric(copyOp, toDelete))
        toDelete.push_back(copyOp.getOperation());
    }
  });
  for (Operation *op : toDelete) op->erase();
}

namespace {
struct LLVMGPUTileAndDistributePass
    : public LLVMGPUTileAndDistributeBase<LLVMGPUTileAndDistributePass> {
 private:
  // Distribute the workloads to warp if true otherwise distribute to threads.
  bool distributeToWarp = false;

 public:
  LLVMGPUTileAndDistributePass(bool distributeToWarp)
      : distributeToWarp(distributeToWarp) {}
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect, gpu::GPUDialect>();
  }
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto funcOp = getOperation();
    if (!isEntryPoint(funcOp)) return;

    // Promote C matrix and propagate the potential  fill producer into the temp
    // allocation. This needs to be done before reduction tiling.
    /*{
      RewritePatternSet promotionPatterns(&getContext());
      populatePromotionPatterns(context, promotionPatterns, contractOpFilter,
                                {2});
      if (failed(applyPatternsAndFoldGreedily(funcOp,
                                              std::move(promotionPatterns)))) {
        return signalPassFailure();
      }
      propagateSharedMemCopy(funcOp);
    }*/

    // Tile again at the workgroup level since reduction dimension were
    // ignored. Dimensions already tiled will be ignore since we tile to the
    // same size.
    if (failed(tileReduction(funcOp))) {
      return signalPassFailure();
    }
    funcOp.dump();
    LLVM_DEBUG({
      llvm::dbgs() << "After tile reductions:";
      funcOp.dump();
    });

    auto workgroupSize = llvm::to_vector<4>(llvm::map_range(
        getEntryPoint(funcOp)->getWorkgroupSize().value(),
        [&](Attribute attr) { return attr.cast<IntegerAttr>().getInt(); }));

    /*int64_t flatWorkgroupSize =
        workgroupSize[0] * workgroupSize[1] * workgroupSize[2];*/
    // Only promote to workgroup size if there are multiple warps.
    //if (flatWorkgroupSize > kWarpSize) {
      //if(0){
      RewritePatternSet promotionPatterns(&getContext());
      if(!clCodegenDisablePromotion){
      populatePromotionPatterns(context, promotionPatterns, contractOpFilter,
                                {0,1});
      }

      if (failed(applyPatternsAndFoldGreedily(funcOp,
                                              std::move(promotionPatterns)))) {
        return signalPassFailure();
      }
      funcOp.dump();
      // Insert barriers before and after copies to workgroup memory and skip
      // insert barriers between back to back copy to workgroup memory.
      OpBuilder builder(&getContext());
      funcOp.walk([&builder](Operation *copyOp) {
        if (hasMarker(copyOp, getCopyToWorkgroupMemoryMarker())) {
          Operation *prevOp = copyOp->getPrevNode();
          if (!prevOp || !hasMarker(prevOp, getCopyToWorkgroupMemoryMarker())) {
            builder.setInsertionPoint(copyOp);
            builder.create<gpu::BarrierOp>(copyOp->getLoc());
          }
          Operation *nextOp = copyOp->getNextNode();
          if (!nextOp || !hasMarker(nextOp, getCopyToWorkgroupMemoryMarker())) {
            builder.setInsertionPointAfter(copyOp);
            builder.create<gpu::BarrierOp>(copyOp->getLoc());
          }
        }
      });
    //}

    {
      RewritePatternSet promotionCanonicalization =
          linalg::getLinalgTilingCanonicalizationPatterns(context);
      if (failed(applyPatternsAndFoldGreedily(
              funcOp, std::move(promotionCanonicalization)))) {
        return signalPassFailure();
      }
    }

    LLVM_DEBUG({
      llvm::dbgs() << "After promotion:";
      funcOp.dump();
    });

    if (distributeToWarp) {
      // Apply last level of tiling and distribute to warps.
      RewritePatternSet warpLevelTilingPatterns(context);
      populateTilingToWarpPatterns(warpLevelTilingPatterns, workgroupSize);
      if (failed(applyPatternsAndFoldGreedily(
              funcOp, std::move(warpLevelTilingPatterns)))) {
        return signalPassFailure();
      }

    } else {
      // Apply last level of tiling and distribute to threads.
      RewritePatternSet threadLevelTilingPatterns(context);
      populateTilingToInvocationPatterns(threadLevelTilingPatterns,
                                         workgroupSize);
      if (failed(applyPatternsAndFoldGreedily(
              funcOp, std::move(threadLevelTilingPatterns)))) {
        return signalPassFailure();
      }
    }
    {
      // Apply canonicalization patterns.
      RewritePatternSet threadTilingCanonicalizationPatterns =
          linalg::getLinalgTilingCanonicalizationPatterns(context);
      populateAffineMinSCFCanonicalizationPattern(
          threadTilingCanonicalizationPatterns);
      if (failed(applyPatternsAndFoldGreedily(
              funcOp, std::move(threadTilingCanonicalizationPatterns)))) {
        return signalPassFailure();
      }
    }

    LLVM_DEBUG({
      llvm::dbgs() << "After tile and distribute to threads:";
      funcOp.dump();
    });
  }
};
}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createLLVMGPUTileAndDistribute(
    bool distributeToWarp) {
  return std::make_unique<LLVMGPUTileAndDistributePass>(distributeToWarp);
}

}  // namespace iree_compiler
}  // namespace mlir

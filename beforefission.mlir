// RUN: iree-opt -pass-pipeline="builtin.module(func.func(iree-codegen-fission-transfer-ops-in-control-flow),cse,canonicalize)" %s --debug-only="iree-codegen-fission-transfer-ops-in-control-flow" --mlir-print-ir-after-failure
func.func @conv_2d_bfloat16_forward_16x24x17x288_nhwc_288x3x3x288_fhwc_nhwf_1x1s_1x1p_1x1d_1g$async_dispatch_0_conv_16x24x17x288x3x3x288_bf16xbf16xf32() attributes {translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false, use_igemm_convolution = true>}>} {
  %c768 = arith.constant 768 : index
  %c512 = arith.constant 512 : index
  %c1088 = arith.constant 1088 : index
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant 0.000000e+00 : bf16
  %c162 = arith.constant 162 : index
  %c2 = arith.constant 2 : index
  %c256 = arith.constant 256 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %cst_1 = arith.constant dense<0.000000e+00> : vector<1x8x1x1x4x1xf32>
  %cst_2 = arith.constant dense<0.000000e+00> : vector<1x1x1x8xbf16>
  %thread_id_x = gpu.thread_id  x upper_bound 256
  %0 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<16x24x17x288xbf16, #hal.descriptor_type<storage_buffer>>
  %1 = amdgpu.fat_raw_buffer_cast %0 resetOffset : memref<16x24x17x288xbf16, #hal.descriptor_type<storage_buffer>> to memref<16x24x17x288xbf16, #amdgpu.address_space<fat_raw_buffer>>
  memref.assume_alignment %1, 64 : memref<16x24x17x288xbf16, #amdgpu.address_space<fat_raw_buffer>>
  %2 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<288x2592xbf16, #hal.descriptor_type<storage_buffer>>
  %3 = amdgpu.fat_raw_buffer_cast %2 resetOffset : memref<288x2592xbf16, #hal.descriptor_type<storage_buffer>> to memref<288x2592xbf16, #amdgpu.address_space<fat_raw_buffer>>
  memref.assume_alignment %3, 64 : memref<288x2592xbf16, #amdgpu.address_space<fat_raw_buffer>>
  %4 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(2) alignment(64) offset(%c0) flags(Indirect) : memref<16x24x17x288xbf16, #hal.descriptor_type<storage_buffer>>
  %5 = amdgpu.fat_raw_buffer_cast %4 resetOffset : memref<16x24x17x288xbf16, #hal.descriptor_type<storage_buffer>> to memref<16x24x17x288xbf16, #amdgpu.address_space<fat_raw_buffer>>
  memref.assume_alignment %5, 64 : memref<16x24x17x288xbf16, #amdgpu.address_space<fat_raw_buffer>>
  scf.forall (%arg0, %arg1, %arg2) in (16, 3, 9) {
    %6 = affine.apply affine_map<(d0) -> (d0 * 32)>(%arg2)
    %7 = affine.apply affine_map<(d0) -> (d0 * 8)>(%arg1)
    %subview = memref.subview %5[%arg0, %7, 0, %6] [1, 8, 17, 32] [1, 1, 1, 1] : memref<16x24x17x288xbf16, #amdgpu.address_space<fat_raw_buffer>> to memref<1x8x17x32xbf16, strided<[117504, 4896, 288, 1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>
    %alloc = memref.alloc() : memref<1x8x32x36xbf16, #gpu.address_space<workgroup>>
    %subview_3 = memref.subview %alloc[0, 0, 0, 0] [1, 8, 32, 32] [1, 1, 1, 1] : memref<1x8x32x36xbf16, #gpu.address_space<workgroup>> to memref<1x8x32x32xbf16, strided<[9216, 1152, 36, 1]>, #gpu.address_space<workgroup>>
    %alloc_4 = memref.alloc() : memref<32x36xbf16, #gpu.address_space<workgroup>>
    %subview_5 = memref.subview %alloc_4[0, 0] [32, 32] [1, 1] : memref<32x36xbf16, #gpu.address_space<workgroup>> to memref<32x32xbf16, strided<[36, 1]>, #gpu.address_space<workgroup>>
    %alloc_6 = memref.alloc() : memref<1x8x32x34xf32, #gpu.address_space<workgroup>>
    %subview_7 = memref.subview %alloc_6[0, 0, 0, 0] [1, 8, 32, 32] [1, 1, 1, 1] : memref<1x8x32x34xf32, #gpu.address_space<workgroup>> to memref<1x8x32x32xf32, strided<[8704, 1088, 34, 1]>, #gpu.address_space<workgroup>>
    %8:2 = affine.delinearize_index %thread_id_x into (4, 64) : index, index
    gpu.barrier
    %9:2 = affine.delinearize_index %8#0 into (2, 2) : index, index
    %10 = affine.linearize_index disjoint [%9#0, %c0] by (2, 16) : index
    %11 = affine.linearize_index disjoint [%9#1, %c0] by (2, 16) : index
    %subview_8 = memref.subview %subview_7[0, 0, %10, %11] [1, 8, 16, 16] [1, 1, 1, 1] : memref<1x8x32x32xf32, strided<[8704, 1088, 34, 1]>, #gpu.address_space<workgroup>> to memref<1x8x16x16xf32, strided<[8704, 1088, 34, 1], offset: ?>, #gpu.address_space<workgroup>>
    %12 = gpu.lane_id upper_bound 64
    %13:3 = affine.delinearize_index %12 into (4, 16) : index, index, index
    %14 = affine.linearize_index disjoint [%13#1, %c0] by (4, 4) : index
    %subview_9 = memref.subview %subview_8[0, 0, %14, %13#2] [1, 8, 4, 1] [1, 1, 1, 1] : memref<1x8x16x16xf32, strided<[8704, 1088, 34, 1], offset: ?>, #gpu.address_space<workgroup>> to memref<1x8x4x1xf32, strided<[8704, 1088, 34, 1], offset: ?>, #gpu.address_space<workgroup>>
    %subview_10 = memref.subview %1[%arg0, 0, 0, 0] [1, 24, 17, 288] [1, 1, 1, 1] : memref<16x24x17x288xbf16, #amdgpu.address_space<fat_raw_buffer>> to memref<1x24x17x288xbf16, strided<[117504, 4896, 288, 1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>
    %15 = affine.apply affine_map<(d0)[s0, s1, s2] -> (d0 + s0 + s1 * 64 + s2 * 128)>(%c0)[%12, %9#1, %9#0]
    %16:3 = affine.delinearize_index %15 into (8, 32, 4) : index, index, index
    %17 = affine.apply affine_map<(d0) -> (d0 * 8)>(%16#2)
    %subview_11 = memref.subview %subview_3[0, %16#0, %16#1, %17] [1, 1, 1, 8] [1, 1, 1, 1] : memref<1x8x32x32xbf16, strided<[9216, 1152, 36, 1]>, #gpu.address_space<workgroup>> to memref<1x1x1x8xbf16, strided<[9216, 1152, 36, 1], offset: ?>, #gpu.address_space<workgroup>>
    %18 = affine.min affine_map<(d0) -> (17, d0)>(%16#1)
    %19 = affine.min affine_map<(d0) -> (-d0 + 17, 1)>(%18)
    %20 = vector.create_mask %c1, %c1, %19, %c8 : vector<1x1x1x8xi1>
    %21 = affine.apply affine_map<(d0)[s0, s1, s2] -> (d0 + s0 + s1 * 64 + s2 * 128)>(%c256)[%12, %9#1, %9#0]
    %22:3 = affine.delinearize_index %21 into (8, 32, 4) : index, index, index
    %23 = affine.apply affine_map<(d0) -> (d0 * 8)>(%22#2)
    %subview_12 = memref.subview %subview_3[0, %22#0, %22#1, %23] [1, 1, 1, 8] [1, 1, 1, 1] : memref<1x8x32x32xbf16, strided<[9216, 1152, 36, 1]>, #gpu.address_space<workgroup>> to memref<1x1x1x8xbf16, strided<[9216, 1152, 36, 1], offset: ?>, #gpu.address_space<workgroup>>
    %24 = affine.min affine_map<(d0) -> (17, d0)>(%22#1)
    %25 = affine.min affine_map<(d0) -> (-d0 + 17, 1)>(%24)
    %26 = vector.create_mask %c1, %c1, %25, %c8 : vector<1x1x1x8xi1>
    %27 = affine.apply affine_map<(d0)[s0, s1, s2] -> (d0 + s0 + s1 * 64 + s2 * 128)>(%c512)[%12, %9#1, %9#0]
    %28:3 = affine.delinearize_index %27 into (8, 32, 4) : index, index, index
    %29 = affine.apply affine_map<(d0) -> (d0 * 8)>(%28#2)
    %subview_13 = memref.subview %subview_3[0, %28#0, %28#1, %29] [1, 1, 1, 8] [1, 1, 1, 1] : memref<1x8x32x32xbf16, strided<[9216, 1152, 36, 1]>, #gpu.address_space<workgroup>> to memref<1x1x1x8xbf16, strided<[9216, 1152, 36, 1], offset: ?>, #gpu.address_space<workgroup>>
    %30 = affine.min affine_map<(d0) -> (17, d0)>(%28#1)
    %31 = affine.min affine_map<(d0) -> (-d0 + 17, 1)>(%30)
    %32 = vector.create_mask %c1, %c1, %31, %c8 : vector<1x1x1x8xi1>
    %33 = affine.apply affine_map<(d0)[s0, s1, s2] -> (d0 + s0 + s1 * 64 + s2 * 128)>(%c768)[%12, %9#1, %9#0]
    %34:3 = affine.delinearize_index %33 into (8, 32, 4) : index, index, index
    %35 = affine.apply affine_map<(d0) -> (d0 * 8)>(%34#2)
    %subview_14 = memref.subview %subview_3[0, %34#0, %34#1, %35] [1, 1, 1, 8] [1, 1, 1, 1] : memref<1x8x32x32xbf16, strided<[9216, 1152, 36, 1]>, #gpu.address_space<workgroup>> to memref<1x1x1x8xbf16, strided<[9216, 1152, 36, 1], offset: ?>, #gpu.address_space<workgroup>>
    %36 = affine.min affine_map<(d0) -> (17, d0)>(%34#1)
    %37 = affine.min affine_map<(d0) -> (-d0 + 17, 1)>(%36)
    %38 = vector.create_mask %c1, %c1, %37, %c8 : vector<1x1x1x8xi1>
    %39 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 * 64 + s2 * 128)>()[%12, %9#1, %9#0]
    %40:2 = affine.delinearize_index %39 into (32, 8) : index, index
    %41 = affine.apply affine_map<()[s0] -> (s0 * 4)>()[%40#1]
    %subview_15 = memref.subview %subview_5[%40#0, %41] [1, 4] [1, 1] : memref<32x32xbf16, strided<[36, 1]>, #gpu.address_space<workgroup>> to memref<1x4xbf16, strided<[36, 1], offset: ?>, #gpu.address_space<workgroup>>
    %42 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + s0)>(%arg2)[%40#0]
    %expand_shape = memref.expand_shape %subview_5 [[0, 1], [2, 3]] output_shape [2, 16, 2, 16] : memref<32x32xbf16, strided<[36, 1]>, #gpu.address_space<workgroup>> into memref<2x16x2x16xbf16, strided<[576, 36, 16, 1]>, #gpu.address_space<workgroup>>
    %expand_shape_16 = memref.expand_shape %subview_3 [[0], [1], [2, 3], [4, 5]] output_shape [1, 8, 2, 16, 2, 16] : memref<1x8x32x32xbf16, strided<[9216, 1152, 36, 1]>, #gpu.address_space<workgroup>> into memref<1x8x2x16x2x16xbf16, strided<[9216, 1152, 576, 36, 16, 1]>, #gpu.address_space<workgroup>>
    %43 = scf.for %arg3 = %c0 to %c162 step %c2 iter_args(%arg4 = %cst_1) -> (vector<1x8x1x1x4x1xf32>) {
      gpu.barrier
      %alloca_18 = memref.alloca(%19) : memref<1x1x?x8xbf16, #gpu.address_space<private>>
      %46 = affine.apply affine_map<(d0, d1) -> (d0 * 16 + d1 * 8)>(%arg3, %16#2)
      %47:3 = affine.delinearize_index %46 into (3, 3, 288) : index, index, index
      scf.for %arg5 = %c0 to %19 step %c1 {
        %subview_23 = memref.subview %alloca_18[0, 0, %arg5, 0] [1, 1, 1, 8] [1, 1, 1, 1] : memref<1x1x?x8xbf16, #gpu.address_space<private>> to memref<1x1x1x8xbf16, strided<[?, ?, 8, 1], offset: ?>, #gpu.address_space<private>>
        %170 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 + d1 + d2 * 17 + d3 * 136)>(%arg5, %18, %16#0, %arg1)
        %171:2 = affine.delinearize_index %170 into (24, 17) : index, index
        %172 = affine.max affine_map<(d0, d1) -> (-d0 - d1 + 1, 0)>(%171#0, %47#0)
        %173 = affine.max affine_map<(d0, d1) -> (0, d0 + d1 - 1)>(%171#0, %47#0)
        %174 = affine.min affine_map<(d0) -> (24, d0)>(%173)
        %175 = affine.min affine_map<(d0, d1) -> (-d0 + 24, -d1 + 1)>(%174, %172)
        %176 = affine.max affine_map<(d0) -> (0, d0)>(%175)
        %177 = affine.max affine_map<(d0, d1) -> (-d0 - d1 + 1, 0)>(%171#1, %47#1)
        %178 = affine.max affine_map<(d0, d1) -> (0, d0 + d1 - 1)>(%171#1, %47#1)
        %179 = affine.min affine_map<(d0) -> (17, d0)>(%178)
        %180 = affine.min affine_map<(d0, d1) -> (-d0 + 17, -d1 + 1)>(%179, %177)
        %181 = affine.max affine_map<(d0) -> (0, d0)>(%180)
        %subview_24 = memref.subview %subview_10[0, %174, %179, %47#2] [1, %176, %181, 8] [1, 1, 1, 1] : memref<1x24x17x288xbf16, strided<[117504, 4896, 288, 1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>> to memref<1x?x?x8xbf16, strided<[117504, 4896, 288, 1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>
        %182 = arith.index_castui %176 : index to i1
        %183 = arith.index_castui %181 : index to i1
        %184 = arith.andi %182, %183 : i1
        %185 = vector.transfer_read %subview_24[%c0, %c0, %c0, %c0], %cst_0 {in_bounds = [true, true, true, true]} : memref<1x?x?x8xbf16, strided<[117504, 4896, 288, 1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1x1x1x8xbf16>
        %186 = arith.select %184, %185, %cst_2 : vector<1x1x1x8xbf16>
        vector.transfer_write %186, %subview_23[%c0, %c0, %c0, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x1x8xbf16>, memref<1x1x1x8xbf16, strided<[?, ?, 8, 1], offset: ?>, #gpu.address_space<private>>
      }
      %48 = vector.transfer_read %alloca_18[%c0, %c0, %c0, %c0], %cst_0, %20 {in_bounds = [true, true, true, true]} : memref<1x1x?x8xbf16, #gpu.address_space<private>>, vector<1x1x1x8xbf16>
      vector.transfer_write %48, %subview_11[%c0, %c0, %c0, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x1x8xbf16>, memref<1x1x1x8xbf16, strided<[9216, 1152, 36, 1], offset: ?>, #gpu.address_space<workgroup>>
      %alloca_19 = memref.alloca(%25) : memref<1x1x?x8xbf16, #gpu.address_space<private>>
      %49 = affine.apply affine_map<(d0, d1) -> (d0 * 16 + d1 * 8)>(%arg3, %22#2)
      %50:3 = affine.delinearize_index %49 into (3, 3, 288) : index, index, index
      scf.for %arg5 = %c0 to %25 step %c1 {
        %subview_23 = memref.subview %alloca_19[0, 0, %arg5, 0] [1, 1, 1, 8] [1, 1, 1, 1] : memref<1x1x?x8xbf16, #gpu.address_space<private>> to memref<1x1x1x8xbf16, strided<[?, ?, 8, 1], offset: ?>, #gpu.address_space<private>>
        %170 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 + d1 + d2 * 17 + d3 * 136)>(%arg5, %24, %22#0, %arg1)
        %171:2 = affine.delinearize_index %170 into (24, 17) : index, index
        %172 = affine.max affine_map<(d0, d1) -> (-d0 - d1 + 1, 0)>(%171#0, %50#0)
        %173 = affine.max affine_map<(d0, d1) -> (0, d0 + d1 - 1)>(%171#0, %50#0)
        %174 = affine.min affine_map<(d0) -> (24, d0)>(%173)
        %175 = affine.min affine_map<(d0, d1) -> (-d0 + 24, -d1 + 1)>(%174, %172)
        %176 = affine.max affine_map<(d0) -> (0, d0)>(%175)
        %177 = affine.max affine_map<(d0, d1) -> (-d0 - d1 + 1, 0)>(%171#1, %50#1)
        %178 = affine.max affine_map<(d0, d1) -> (0, d0 + d1 - 1)>(%171#1, %50#1)
        %179 = affine.min affine_map<(d0) -> (17, d0)>(%178)
        %180 = affine.min affine_map<(d0, d1) -> (-d0 + 17, -d1 + 1)>(%179, %177)
        %181 = affine.max affine_map<(d0) -> (0, d0)>(%180)
        %subview_24 = memref.subview %subview_10[0, %174, %179, %50#2] [1, %176, %181, 8] [1, 1, 1, 1] : memref<1x24x17x288xbf16, strided<[117504, 4896, 288, 1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>> to memref<1x?x?x8xbf16, strided<[117504, 4896, 288, 1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>
        %182 = arith.index_castui %176 : index to i1
        %183 = arith.index_castui %181 : index to i1
        %184 = arith.andi %182, %183 : i1
        %185 = vector.transfer_read %subview_24[%c0, %c0, %c0, %c0], %cst_0 {in_bounds = [true, true, true, true]} : memref<1x?x?x8xbf16, strided<[117504, 4896, 288, 1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1x1x1x8xbf16>
        %186 = arith.select %184, %185, %cst_2 : vector<1x1x1x8xbf16>
        vector.transfer_write %186, %subview_23[%c0, %c0, %c0, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x1x8xbf16>, memref<1x1x1x8xbf16, strided<[?, ?, 8, 1], offset: ?>, #gpu.address_space<private>>
      }
      %51 = vector.transfer_read %alloca_19[%c0, %c0, %c0, %c0], %cst_0, %26 {in_bounds = [true, true, true, true]} : memref<1x1x?x8xbf16, #gpu.address_space<private>>, vector<1x1x1x8xbf16>
      vector.transfer_write %51, %subview_12[%c0, %c0, %c0, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x1x8xbf16>, memref<1x1x1x8xbf16, strided<[9216, 1152, 36, 1], offset: ?>, #gpu.address_space<workgroup>>
      %alloca_20 = memref.alloca(%31) : memref<1x1x?x8xbf16, #gpu.address_space<private>>
      %52 = affine.apply affine_map<(d0, d1) -> (d0 * 16 + d1 * 8)>(%arg3, %28#2)
      %53:3 = affine.delinearize_index %52 into (3, 3, 288) : index, index, index
      scf.for %arg5 = %c0 to %31 step %c1 {
        %subview_23 = memref.subview %alloca_20[0, 0, %arg5, 0] [1, 1, 1, 8] [1, 1, 1, 1] : memref<1x1x?x8xbf16, #gpu.address_space<private>> to memref<1x1x1x8xbf16, strided<[?, ?, 8, 1], offset: ?>, #gpu.address_space<private>>
        %170 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 + d1 + d2 * 17 + d3 * 136)>(%arg5, %30, %28#0, %arg1)
        %171:2 = affine.delinearize_index %170 into (24, 17) : index, index
        %172 = affine.max affine_map<(d0, d1) -> (-d0 - d1 + 1, 0)>(%171#0, %53#0)
        %173 = affine.max affine_map<(d0, d1) -> (0, d0 + d1 - 1)>(%171#0, %53#0)
        %174 = affine.min affine_map<(d0) -> (24, d0)>(%173)
        %175 = affine.min affine_map<(d0, d1) -> (-d0 + 24, -d1 + 1)>(%174, %172)
        %176 = affine.max affine_map<(d0) -> (0, d0)>(%175)
        %177 = affine.max affine_map<(d0, d1) -> (-d0 - d1 + 1, 0)>(%171#1, %53#1)
        %178 = affine.max affine_map<(d0, d1) -> (0, d0 + d1 - 1)>(%171#1, %53#1)
        %179 = affine.min affine_map<(d0) -> (17, d0)>(%178)
        %180 = affine.min affine_map<(d0, d1) -> (-d0 + 17, -d1 + 1)>(%179, %177)
        %181 = affine.max affine_map<(d0) -> (0, d0)>(%180)
        %subview_24 = memref.subview %subview_10[0, %174, %179, %53#2] [1, %176, %181, 8] [1, 1, 1, 1] : memref<1x24x17x288xbf16, strided<[117504, 4896, 288, 1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>> to memref<1x?x?x8xbf16, strided<[117504, 4896, 288, 1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>
        %182 = arith.index_castui %176 : index to i1
        %183 = arith.index_castui %181 : index to i1
        %184 = arith.andi %182, %183 : i1
        %185 = vector.transfer_read %subview_24[%c0, %c0, %c0, %c0], %cst_0 {in_bounds = [true, true, true, true]} : memref<1x?x?x8xbf16, strided<[117504, 4896, 288, 1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1x1x1x8xbf16>
        %186 = arith.select %184, %185, %cst_2 : vector<1x1x1x8xbf16>
        vector.transfer_write %186, %subview_23[%c0, %c0, %c0, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x1x8xbf16>, memref<1x1x1x8xbf16, strided<[?, ?, 8, 1], offset: ?>, #gpu.address_space<private>>
      }
      %54 = vector.transfer_read %alloca_20[%c0, %c0, %c0, %c0], %cst_0, %32 {in_bounds = [true, true, true, true]} : memref<1x1x?x8xbf16, #gpu.address_space<private>>, vector<1x1x1x8xbf16>
      vector.transfer_write %54, %subview_13[%c0, %c0, %c0, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x1x8xbf16>, memref<1x1x1x8xbf16, strided<[9216, 1152, 36, 1], offset: ?>, #gpu.address_space<workgroup>>
      %alloca_21 = memref.alloca(%37) : memref<1x1x?x8xbf16, #gpu.address_space<private>>
      %55 = affine.apply affine_map<(d0, d1) -> (d0 * 16 + d1 * 8)>(%arg3, %34#2)
      %56:3 = affine.delinearize_index %55 into (3, 3, 288) : index, index, index
      scf.for %arg5 = %c0 to %37 step %c1 {
        %subview_23 = memref.subview %alloca_21[0, 0, %arg5, 0] [1, 1, 1, 8] [1, 1, 1, 1] : memref<1x1x?x8xbf16, #gpu.address_space<private>> to memref<1x1x1x8xbf16, strided<[?, ?, 8, 1], offset: ?>, #gpu.address_space<private>>
        %170 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 + d1 + d2 * 17 + d3 * 136)>(%arg5, %36, %34#0, %arg1)
        %171:2 = affine.delinearize_index %170 into (24, 17) : index, index
        %172 = affine.max affine_map<(d0, d1) -> (-d0 - d1 + 1, 0)>(%171#0, %56#0)
        %173 = affine.max affine_map<(d0, d1) -> (0, d0 + d1 - 1)>(%171#0, %56#0)
        %174 = affine.min affine_map<(d0) -> (24, d0)>(%173)
        %175 = affine.min affine_map<(d0, d1) -> (-d0 + 24, -d1 + 1)>(%174, %172)
        %176 = affine.max affine_map<(d0) -> (0, d0)>(%175)
        %177 = affine.max affine_map<(d0, d1) -> (-d0 - d1 + 1, 0)>(%171#1, %56#1)
        %178 = affine.max affine_map<(d0, d1) -> (0, d0 + d1 - 1)>(%171#1, %56#1)
        %179 = affine.min affine_map<(d0) -> (17, d0)>(%178)
        %180 = affine.min affine_map<(d0, d1) -> (-d0 + 17, -d1 + 1)>(%179, %177)
        %181 = affine.max affine_map<(d0) -> (0, d0)>(%180)
        %subview_24 = memref.subview %subview_10[0, %174, %179, %56#2] [1, %176, %181, 8] [1, 1, 1, 1] : memref<1x24x17x288xbf16, strided<[117504, 4896, 288, 1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>> to memref<1x?x?x8xbf16, strided<[117504, 4896, 288, 1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>
        %182 = arith.index_castui %176 : index to i1
        %183 = arith.index_castui %181 : index to i1
        %184 = arith.andi %182, %183 : i1
        %185 = vector.transfer_read %subview_24[%c0, %c0, %c0, %c0], %cst_0 {in_bounds = [true, true, true, true]} : memref<1x?x?x8xbf16, strided<[117504, 4896, 288, 1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1x1x1x8xbf16>
        %186 = arith.select %184, %185, %cst_2 : vector<1x1x1x8xbf16>
        vector.transfer_write %186, %subview_23[%c0, %c0, %c0, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x1x8xbf16>, memref<1x1x1x8xbf16, strided<[?, ?, 8, 1], offset: ?>, #gpu.address_space<private>>
      }
      %57 = vector.transfer_read %alloca_21[%c0, %c0, %c0, %c0], %cst_0, %38 {in_bounds = [true, true, true, true]} : memref<1x1x?x8xbf16, #gpu.address_space<private>>, vector<1x1x1x8xbf16>
      vector.transfer_write %57, %subview_14[%c0, %c0, %c0, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x1x8xbf16>, memref<1x1x1x8xbf16, strided<[9216, 1152, 36, 1], offset: ?>, #gpu.address_space<workgroup>>
      %58 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + s0 * 4)>(%arg3)[%40#1]
      %subview_22 = memref.subview %3[%42, %58] [1, 4] [1, 1] : memref<288x2592xbf16, #amdgpu.address_space<fat_raw_buffer>> to memref<1x4xbf16, strided<[2592, 1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>
      %59 = vector.transfer_read %subview_22[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<1x4xbf16, strided<[2592, 1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, vector<1x4xbf16>
      vector.transfer_write %59, %subview_15[%c0, %c0] {in_bounds = [true, true]} : vector<1x4xbf16>, memref<1x4xbf16, strided<[36, 1], offset: ?>, #gpu.address_space<workgroup>>
      gpu.barrier
      %60 = vector.transfer_read %expand_shape_16[%c0, %c0, %9#0, %13#2, %c0, %14], %cst_0 {in_bounds = [true, true, true, true, true, true]} : memref<1x8x2x16x2x16xbf16, strided<[9216, 1152, 576, 36, 16, 1]>, #gpu.address_space<workgroup>>, vector<1x8x1x1x2x4xbf16>
      %61 = vector.transpose %60, [0, 1, 2, 4, 3, 5] : vector<1x8x1x1x2x4xbf16> to vector<1x8x1x2x1x4xbf16>
      %62 = vector.transfer_read %expand_shape[%9#1, %13#2, %c0, %14], %cst_0 {in_bounds = [true, true, true, true]} : memref<2x16x2x16xbf16, strided<[576, 36, 16, 1]>, #gpu.address_space<workgroup>>, vector<1x1x2x4xbf16>
      %63 = vector.transpose %62, [0, 2, 1, 3] : vector<1x1x2x4xbf16> to vector<1x2x1x4xbf16>
      %64 = vector.extract %61[0, 0, 0, 0] : vector<1x4xbf16> from vector<1x8x1x2x1x4xbf16>
      %65 = vector.extract %63[0, 0] : vector<1x4xbf16> from vector<1x2x1x4xbf16>
      %66 = vector.extract %arg4[0, 0, 0, 0] : vector<4x1xf32> from vector<1x8x1x1x4x1xf32>
      %67 = vector.shape_cast %64 : vector<1x4xbf16> to vector<4xbf16>
      %68 = vector.shape_cast %65 : vector<1x4xbf16> to vector<4xbf16>
      %69 = vector.shape_cast %66 : vector<4x1xf32> to vector<4xf32>
      %70 = amdgpu.mfma %67 * %68 + %69 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xbf16>, vector<4xbf16>, vector<4xf32>
      %71 = vector.extract %61[0, 1, 0, 0] : vector<1x4xbf16> from vector<1x8x1x2x1x4xbf16>
      %72 = vector.extract %arg4[0, 1, 0, 0] : vector<4x1xf32> from vector<1x8x1x1x4x1xf32>
      %73 = vector.shape_cast %71 : vector<1x4xbf16> to vector<4xbf16>
      %74 = vector.shape_cast %65 : vector<1x4xbf16> to vector<4xbf16>
      %75 = vector.shape_cast %72 : vector<4x1xf32> to vector<4xf32>
      %76 = amdgpu.mfma %73 * %74 + %75 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xbf16>, vector<4xbf16>, vector<4xf32>
      %77 = vector.extract %61[0, 2, 0, 0] : vector<1x4xbf16> from vector<1x8x1x2x1x4xbf16>
      %78 = vector.extract %arg4[0, 2, 0, 0] : vector<4x1xf32> from vector<1x8x1x1x4x1xf32>
      %79 = vector.shape_cast %77 : vector<1x4xbf16> to vector<4xbf16>
      %80 = vector.shape_cast %65 : vector<1x4xbf16> to vector<4xbf16>
      %81 = vector.shape_cast %78 : vector<4x1xf32> to vector<4xf32>
      %82 = amdgpu.mfma %79 * %80 + %81 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xbf16>, vector<4xbf16>, vector<4xf32>
      %83 = vector.extract %61[0, 3, 0, 0] : vector<1x4xbf16> from vector<1x8x1x2x1x4xbf16>
      %84 = vector.extract %arg4[0, 3, 0, 0] : vector<4x1xf32> from vector<1x8x1x1x4x1xf32>
      %85 = vector.shape_cast %83 : vector<1x4xbf16> to vector<4xbf16>
      %86 = vector.shape_cast %65 : vector<1x4xbf16> to vector<4xbf16>
      %87 = vector.shape_cast %84 : vector<4x1xf32> to vector<4xf32>
      %88 = amdgpu.mfma %85 * %86 + %87 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xbf16>, vector<4xbf16>, vector<4xf32>
      %89 = vector.extract %61[0, 4, 0, 0] : vector<1x4xbf16> from vector<1x8x1x2x1x4xbf16>
      %90 = vector.extract %arg4[0, 4, 0, 0] : vector<4x1xf32> from vector<1x8x1x1x4x1xf32>
      %91 = vector.shape_cast %89 : vector<1x4xbf16> to vector<4xbf16>
      %92 = vector.shape_cast %65 : vector<1x4xbf16> to vector<4xbf16>
      %93 = vector.shape_cast %90 : vector<4x1xf32> to vector<4xf32>
      %94 = amdgpu.mfma %91 * %92 + %93 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xbf16>, vector<4xbf16>, vector<4xf32>
      %95 = vector.extract %61[0, 5, 0, 0] : vector<1x4xbf16> from vector<1x8x1x2x1x4xbf16>
      %96 = vector.extract %arg4[0, 5, 0, 0] : vector<4x1xf32> from vector<1x8x1x1x4x1xf32>
      %97 = vector.shape_cast %95 : vector<1x4xbf16> to vector<4xbf16>
      %98 = vector.shape_cast %65 : vector<1x4xbf16> to vector<4xbf16>
      %99 = vector.shape_cast %96 : vector<4x1xf32> to vector<4xf32>
      %100 = amdgpu.mfma %97 * %98 + %99 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xbf16>, vector<4xbf16>, vector<4xf32>
      %101 = vector.extract %61[0, 6, 0, 0] : vector<1x4xbf16> from vector<1x8x1x2x1x4xbf16>
      %102 = vector.extract %arg4[0, 6, 0, 0] : vector<4x1xf32> from vector<1x8x1x1x4x1xf32>
      %103 = vector.shape_cast %101 : vector<1x4xbf16> to vector<4xbf16>
      %104 = vector.shape_cast %65 : vector<1x4xbf16> to vector<4xbf16>
      %105 = vector.shape_cast %102 : vector<4x1xf32> to vector<4xf32>
      %106 = amdgpu.mfma %103 * %104 + %105 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xbf16>, vector<4xbf16>, vector<4xf32>
      %107 = vector.extract %61[0, 7, 0, 0] : vector<1x4xbf16> from vector<1x8x1x2x1x4xbf16>
      %108 = vector.extract %arg4[0, 7, 0, 0] : vector<4x1xf32> from vector<1x8x1x1x4x1xf32>
      %109 = vector.shape_cast %107 : vector<1x4xbf16> to vector<4xbf16>
      %110 = vector.shape_cast %65 : vector<1x4xbf16> to vector<4xbf16>
      %111 = vector.shape_cast %108 : vector<4x1xf32> to vector<4xf32>
      %112 = amdgpu.mfma %109 * %110 + %111 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xbf16>, vector<4xbf16>, vector<4xf32>
      %113 = vector.extract %61[0, 0, 0, 1] : vector<1x4xbf16> from vector<1x8x1x2x1x4xbf16>
      %114 = vector.extract %63[0, 1] : vector<1x4xbf16> from vector<1x2x1x4xbf16>
      %115 = vector.shape_cast %113 : vector<1x4xbf16> to vector<4xbf16>
      %116 = vector.shape_cast %114 : vector<1x4xbf16> to vector<4xbf16>
      %117 = amdgpu.mfma %115 * %116 + %70 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xbf16>, vector<4xbf16>, vector<4xf32>
      %118 = vector.shape_cast %117 : vector<4xf32> to vector<4x1xf32>
      %119 = vector.broadcast %118 : vector<4x1xf32> to vector<1x1x1x1x4x1xf32>
      %120 = vector.extract %61[0, 1, 0, 1] : vector<1x4xbf16> from vector<1x8x1x2x1x4xbf16>
      %121 = vector.shape_cast %120 : vector<1x4xbf16> to vector<4xbf16>
      %122 = vector.shape_cast %114 : vector<1x4xbf16> to vector<4xbf16>
      %123 = amdgpu.mfma %121 * %122 + %76 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xbf16>, vector<4xbf16>, vector<4xf32>
      %124 = vector.shape_cast %123 : vector<4xf32> to vector<4x1xf32>
      %125 = vector.broadcast %124 : vector<4x1xf32> to vector<1x1x1x1x4x1xf32>
      %126 = vector.extract %61[0, 2, 0, 1] : vector<1x4xbf16> from vector<1x8x1x2x1x4xbf16>
      %127 = vector.shape_cast %126 : vector<1x4xbf16> to vector<4xbf16>
      %128 = vector.shape_cast %114 : vector<1x4xbf16> to vector<4xbf16>
      %129 = amdgpu.mfma %127 * %128 + %82 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xbf16>, vector<4xbf16>, vector<4xf32>
      %130 = vector.shape_cast %129 : vector<4xf32> to vector<4x1xf32>
      %131 = vector.broadcast %130 : vector<4x1xf32> to vector<1x1x1x1x4x1xf32>
      %132 = vector.extract %61[0, 3, 0, 1] : vector<1x4xbf16> from vector<1x8x1x2x1x4xbf16>
      %133 = vector.shape_cast %132 : vector<1x4xbf16> to vector<4xbf16>
      %134 = vector.shape_cast %114 : vector<1x4xbf16> to vector<4xbf16>
      %135 = amdgpu.mfma %133 * %134 + %88 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xbf16>, vector<4xbf16>, vector<4xf32>
      %136 = vector.shape_cast %135 : vector<4xf32> to vector<4x1xf32>
      %137 = vector.broadcast %136 : vector<4x1xf32> to vector<1x1x1x1x4x1xf32>
      %138 = vector.extract %61[0, 4, 0, 1] : vector<1x4xbf16> from vector<1x8x1x2x1x4xbf16>
      %139 = vector.shape_cast %138 : vector<1x4xbf16> to vector<4xbf16>
      %140 = vector.shape_cast %114 : vector<1x4xbf16> to vector<4xbf16>
      %141 = amdgpu.mfma %139 * %140 + %94 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xbf16>, vector<4xbf16>, vector<4xf32>
      %142 = vector.shape_cast %141 : vector<4xf32> to vector<4x1xf32>
      %143 = vector.broadcast %142 : vector<4x1xf32> to vector<1x1x1x1x4x1xf32>
      %144 = vector.extract %61[0, 5, 0, 1] : vector<1x4xbf16> from vector<1x8x1x2x1x4xbf16>
      %145 = vector.shape_cast %144 : vector<1x4xbf16> to vector<4xbf16>
      %146 = vector.shape_cast %114 : vector<1x4xbf16> to vector<4xbf16>
      %147 = amdgpu.mfma %145 * %146 + %100 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xbf16>, vector<4xbf16>, vector<4xf32>
      %148 = vector.shape_cast %147 : vector<4xf32> to vector<4x1xf32>
      %149 = vector.broadcast %148 : vector<4x1xf32> to vector<1x1x1x1x4x1xf32>
      %150 = vector.extract %61[0, 6, 0, 1] : vector<1x4xbf16> from vector<1x8x1x2x1x4xbf16>
      %151 = vector.shape_cast %150 : vector<1x4xbf16> to vector<4xbf16>
      %152 = vector.shape_cast %114 : vector<1x4xbf16> to vector<4xbf16>
      %153 = amdgpu.mfma %151 * %152 + %106 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xbf16>, vector<4xbf16>, vector<4xf32>
      %154 = vector.shape_cast %153 : vector<4xf32> to vector<4x1xf32>
      %155 = vector.broadcast %154 : vector<4x1xf32> to vector<1x1x1x1x4x1xf32>
      %156 = vector.extract %61[0, 7, 0, 1] : vector<1x4xbf16> from vector<1x8x1x2x1x4xbf16>
      %157 = vector.shape_cast %156 : vector<1x4xbf16> to vector<4xbf16>
      %158 = vector.shape_cast %114 : vector<1x4xbf16> to vector<4xbf16>
      %159 = amdgpu.mfma %157 * %158 + %112 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xbf16>, vector<4xbf16>, vector<4xf32>
      %160 = vector.shape_cast %159 : vector<4xf32> to vector<4x1xf32>
      %161 = vector.broadcast %160 : vector<4x1xf32> to vector<1x1x1x1x4x1xf32>
      %162 = vector.insert_strided_slice %119, %cst_1 {offsets = [0, 0, 0, 0, 0, 0], strides = [1, 1, 1, 1, 1, 1]} : vector<1x1x1x1x4x1xf32> into vector<1x8x1x1x4x1xf32>
      %163 = vector.insert_strided_slice %125, %162 {offsets = [0, 1, 0, 0, 0, 0], strides = [1, 1, 1, 1, 1, 1]} : vector<1x1x1x1x4x1xf32> into vector<1x8x1x1x4x1xf32>
      %164 = vector.insert_strided_slice %131, %163 {offsets = [0, 2, 0, 0, 0, 0], strides = [1, 1, 1, 1, 1, 1]} : vector<1x1x1x1x4x1xf32> into vector<1x8x1x1x4x1xf32>
      %165 = vector.insert_strided_slice %137, %164 {offsets = [0, 3, 0, 0, 0, 0], strides = [1, 1, 1, 1, 1, 1]} : vector<1x1x1x1x4x1xf32> into vector<1x8x1x1x4x1xf32>
      %166 = vector.insert_strided_slice %143, %165 {offsets = [0, 4, 0, 0, 0, 0], strides = [1, 1, 1, 1, 1, 1]} : vector<1x1x1x1x4x1xf32> into vector<1x8x1x1x4x1xf32>
      %167 = vector.insert_strided_slice %149, %166 {offsets = [0, 5, 0, 0, 0, 0], strides = [1, 1, 1, 1, 1, 1]} : vector<1x1x1x1x4x1xf32> into vector<1x8x1x1x4x1xf32>
      %168 = vector.insert_strided_slice %155, %167 {offsets = [0, 6, 0, 0, 0, 0], strides = [1, 1, 1, 1, 1, 1]} : vector<1x1x1x1x4x1xf32> into vector<1x8x1x1x4x1xf32>
      %169 = vector.insert_strided_slice %161, %168 {offsets = [0, 7, 0, 0, 0, 0], strides = [1, 1, 1, 1, 1, 1]} : vector<1x1x1x1x4x1xf32> into vector<1x8x1x1x4x1xf32>
      scf.yield %169 : vector<1x8x1x1x4x1xf32>
    }
    %alloca = memref.alloca() : memref<1x8x1x4x1x1xf32, #gpu.address_space<private>>
    %44 = vector.transpose %43, [0, 1, 2, 4, 3, 5] : vector<1x8x1x1x4x1xf32> to vector<1x8x1x4x1x1xf32>
    vector.transfer_write %44, %alloca[%c0, %c0, %c0, %c0, %c0, %c0] {in_bounds = [true, true, true, true, true, true]} : vector<1x8x1x4x1x1xf32>, memref<1x8x1x4x1x1xf32, #gpu.address_space<private>>
    %expand_shape_17 = memref.expand_shape %subview_9 [[0], [1], [2, 3], [4, 5]] output_shape [1, 8, 1, 4, 1, 1] : memref<1x8x4x1xf32, strided<[8704, 1088, 34, 1], offset: ?>, #gpu.address_space<workgroup>> into memref<1x8x1x4x1x1xf32, strided<[8704, 1088, 136, 34, 1, 1], offset: ?>, #gpu.address_space<workgroup>>
    %45 = vector.transfer_read %alloca[%c0, %c0, %c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true, true, true]} : memref<1x8x1x4x1x1xf32, #gpu.address_space<private>>, vector<1x8x1x4x1x1xf32>
    vector.transfer_write %45, %expand_shape_17[%c0, %c0, %c0, %c0, %c0, %c0] {in_bounds = [true, true, true, true, true, true]} : vector<1x8x1x4x1x1xf32>, memref<1x8x1x4x1x1xf32, strided<[8704, 1088, 136, 34, 1, 1], offset: ?>, #gpu.address_space<workgroup>>
    gpu.barrier
    scf.for %arg3 = %thread_id_x to %c1088 step %c256 {
      %46:3 = affine.delinearize_index %arg3 into (8, 17, 8) : index, index, index
      %47 = affine.apply affine_map<(d0) -> (d0 * 4)>(%46#2)
      %subview_18 = memref.subview %subview[0, %46#0, %46#1, %47] [1, 1, 1, 4] [1, 1, 1, 1] : memref<1x8x17x32xbf16, strided<[117504, 4896, 288, 1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>> to memref<1x1x1x4xbf16, strided<[117504, 4896, 288, 1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>
      %subview_19 = memref.subview %subview_7[0, %46#0, %46#1, %47] [1, 1, 1, 4] [1, 1, 1, 1] : memref<1x8x32x32xf32, strided<[8704, 1088, 34, 1]>, #gpu.address_space<workgroup>> to memref<1x1x1x4xf32, strided<[8704, 1088, 34, 1], offset: ?>, #gpu.address_space<workgroup>>
      %alloca_20 = memref.alloca() : memref<1x1x1x4xf32, #gpu.address_space<private>>
      %48 = vector.transfer_read %subview_19[%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : memref<1x1x1x4xf32, strided<[8704, 1088, 34, 1], offset: ?>, #gpu.address_space<workgroup>>, vector<1x1x1x4xf32>
      vector.transfer_write %48, %alloca_20[%c0, %c0, %c0, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x1x4xf32>, memref<1x1x1x4xf32, #gpu.address_space<private>>
      %49 = vector.transfer_read %alloca_20[%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : memref<1x1x1x4xf32, #gpu.address_space<private>>, vector<1x1x1x4xf32>
      %50 = arith.truncf %49 : vector<1x1x1x4xf32> to vector<1x1x1x4xbf16>
      vector.transfer_write %50, %subview_18[%c0, %c0, %c0, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x1x4xbf16>, memref<1x1x1x4xbf16, strided<[117504, 4896, 288, 1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>
    }
    gpu.barrier
  } {mapping = [#iree_codegen.workgroup_mapping<z>, #iree_codegen.workgroup_mapping<y>, #iree_codegen.workgroup_mapping<x>]}
  return
}


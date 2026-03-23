#!/usr/bin/env python3
# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""iree_generated_e2e_matmul_test generator for batch matmul tests."""

from typing import Optional
import dataclasses
import typing

from tests.e2e.matmul.common import *
from tests.e2e.matmul.compilation_info import *


# Describes the shape of a batch matmul: batch x m x k @ batch x k x n
@dataclasses.dataclass
class BatchTestShape:
    batch: int
    m: int
    k: int
    n: int
    accumulate: bool


def generate_function_name_batch(
    lhs_rhs_type: MatrixElemTypeId,
    acc_type: MatrixElemTypeId,
    shapes: TestInputMatricesShapes,
    batch_size: int,
    accumulate: bool,
    compilation_info: typing.Optional[CompilationInfo] = None,
):
    input_t = lhs_rhs_type.value
    acc_t = acc_type.value
    lhs_b = batch_size
    lhs_r = int_or_DYN(shapes.lhs_rows)
    lhs_c = int_or_DYN(shapes.lhs_cols)
    rhs_b = batch_size
    rhs_r = int_or_DYN(shapes.rhs_rows)
    rhs_c = int_or_DYN(shapes.rhs_cols)
    acc_b = batch_size
    acc_r = int_or_DYN(shapes.acc_rows)
    acc_c = int_or_DYN(shapes.acc_cols)

    info = ""
    if compilation_info:
        pipeline_name = compilation_info.dispatch_lowering_pass_pipeline
        # Strip #iree_gpu.pipeline<...> wrapper for use in identifiers.
        if pipeline_name.startswith("#iree_gpu.pipeline<"):
            pipeline_name = pipeline_name[len("#iree_gpu.pipeline<") : -1]
        info = f"_for_{pipeline_name}"

    matmul_kind = "batch_matmul_accumulate" if accumulate else "batch_matmul"

    return (
        f"{matmul_kind}_{lhs_b}x{lhs_r}x{lhs_c}x{input_t}_times_"
        + f"{rhs_b}x{rhs_r}x{rhs_c}x{input_t}_into_{acc_b}x{acc_r}x{acc_c}x{acc_t}{info}"
    )


def generate_function_batch(
    lhs_rhs_type: MatrixElemTypeId,
    acc_type: MatrixElemTypeId,
    batch_shape: BatchTestShape,
    transpose_rhs: bool,
    dynamicities: tuple[Dynamicity, Dynamicity, Dynamicity, Dynamicity],
    compilation_info: Optional[CompilationInfo] = None,
):
    # Convert BatchTestShape to TestShape for shape generation
    shape = TestShape(
        m=batch_shape.m,
        k=batch_shape.k,
        n=batch_shape.n,
        accumulate=batch_shape.accumulate,
    )
    shapes = generate_shapes(shape, transpose_rhs, dynamicities[1:])  # Skip batch

    batch_size = batch_shape.batch
    func_name = generate_function_name_batch(
        lhs_rhs_type=lhs_rhs_type,
        acc_type=acc_type,
        shapes=shapes,
        batch_size=batch_size,
        accumulate=batch_shape.accumulate,
        compilation_info=compilation_info,
    )

    # Handle batch dimension dynamicity
    dynamicity_batch = dynamicities[0]
    batch_dim = batch_size if dynamicity_batch == Dynamicity.STATIC else "?"

    lhs_r = int_or_question_mark(shapes.lhs_rows)
    lhs_c = int_or_question_mark(shapes.lhs_cols)
    rhs_r = int_or_question_mark(shapes.rhs_rows)
    rhs_c = int_or_question_mark(shapes.rhs_cols)
    acc_r = int_or_question_mark(shapes.acc_rows)
    acc_c = int_or_question_mark(shapes.acc_cols)

    lhs_tensor_type = f"tensor<{batch_dim}x{lhs_r}x{lhs_c}x{lhs_rhs_type.value}>"
    rhs_tensor_type = f"tensor<{batch_dim}x{rhs_r}x{rhs_c}x{lhs_rhs_type.value}>"
    acc_tensor_type = f"tensor<{batch_dim}x{acc_r}x{acc_c}x{acc_type.value}>"

    (
        compilation_info_string,
        compilation_info_attr,
    ) = generate_compilation_info_string_and_attr(compilation_info)

    args = [("%lhs", lhs_tensor_type), ("%rhs", rhs_tensor_type)]
    if batch_shape.accumulate:
        args += [("%acc", acc_tensor_type)]

    func_definition = compilation_info_string + (
        f"util.func @{func_name}("
        + ", ".join([name + ": " + ty for name, ty in args])
        + f") -> {acc_tensor_type} {{\n"
    )

    if not batch_shape.accumulate:
        literal_zero_for_acc_type = "0.0" if "f" in acc_type.value else "0"
        # Handle dynamic dimensions for empty tensor
        dynamic_dims = []
        if batch_dim == "?":
            func_definition += f"  %c0 = arith.constant 0 : index\n"
            func_definition += (
                f"  %batch_dim = tensor.dim %lhs, %c0 : {lhs_tensor_type}\n"
            )
            dynamic_dims.append("%batch_dim")
        if acc_r == "?":
            func_definition += f"  %c1 = arith.constant 1 : index\n"
            func_definition += f"  %acc_dim1 = tensor.dim %lhs, %c1 : {lhs_tensor_type}\n"
            dynamic_dims.append("%acc_dim1")
        if acc_c == "?":
            func_definition += f"  %c2 = arith.constant 2 : index\n"
            func_definition += f"  %acc_dim2 = tensor.dim %rhs, %c2 : {rhs_tensor_type}\n"
            dynamic_dims.append("%acc_dim2")

        if dynamic_dims:
            func_definition += f"  %init_acc = tensor.empty({', '.join(dynamic_dims)}) : {acc_tensor_type}\n"
        else:
            func_definition += f"  %init_acc = tensor.empty() : {acc_tensor_type}\n"

        func_definition += (
            f"  %c0_acc_type = arith.constant {literal_zero_for_acc_type}: {acc_type.value}\n"
            f"  %acc = linalg.fill ins(%c0_acc_type : {acc_type.value}) outs(%init_acc : {acc_tensor_type}) -> {acc_tensor_type}\n"
        )

    # Generate linalg.batch_matmul with optional transpose
    indexing_maps_attr = ""
    if transpose_rhs:
        indexing_maps_attr = "indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>]"

    func_definition += (
        f"  %result = linalg.batch_matmul {indexing_maps_attr} {compilation_info_attr} ins(%lhs, %rhs: {lhs_tensor_type}, {rhs_tensor_type}) outs(%acc: {acc_tensor_type}) -> {acc_tensor_type}\n"
        f"  util.return %result: {acc_tensor_type}\n"
        f"}}\n"
    )

    signature = ", ".join([ty for name, ty in args]) + f" -> {acc_tensor_type}"
    import_declaration = (
        f"util.func private @module.{func_name}("
        + ", ".join([name + ": !hal.buffer_view" for name, ty in args])
        + ") -> !hal.buffer_view"
    )

    return MLIRFunction(
        name=func_name,
        signature=signature,
        import_declaration=import_declaration,
        definition=func_definition,
    )


call_id_batch = 0


def generate_call_batch(
    function: MLIRFunction,
    lhs_rhs_type: MatrixElemTypeId,
    acc_type: MatrixElemTypeId,
    batch_shape: BatchTestShape,
    transpose_rhs: bool,
):
    global call_id_batch
    func_name = f"{function.name}_{batch_shape.batch}_{batch_shape.m}_{batch_shape.k}_{batch_shape.n}"
    if batch_shape.accumulate:
        func_name = f"{func_name}_acc"
    func_name = f"{func_name}_{call_id_batch}"
    call_id_batch = call_id_batch + 1

    description = f"Batch Matmul shape (BxMxKxN): {batch_shape.batch}x{batch_shape.m}x{batch_shape.k}x{batch_shape.n}"
    op = (
        f"util.func @{func_name}() attributes {{\n"
        f'  iree.reflection = {{description = "{description}"}}\n'
        "} {\n"
        "  %device_index = arith.constant 0 : index\n"
        "  %device = hal.devices.get %device_index : !hal.device\n"
    )

    lhs_shape = [batch_shape.batch, batch_shape.m, batch_shape.k]
    if transpose_rhs:
        rhs_shape = [batch_shape.batch, batch_shape.n, batch_shape.k]
        transpose_rhs_val = 1
    else:
        rhs_shape = [batch_shape.batch, batch_shape.k, batch_shape.n]
        transpose_rhs_val = 0

    matmul_args = [
        ("lhs", lhs_shape, lhs_rhs_type),
        ("rhs", rhs_shape, lhs_rhs_type),
    ]
    check_args = matmul_args.copy()

    if batch_shape.accumulate:
        matmul_args += [
            ("acc", [batch_shape.batch, batch_shape.m, batch_shape.n], acc_type)
        ]
    else:
        op += "  %acc = util.null : !hal.buffer_view\n"

    for arg_name, arg_shape, arg_elemtype in matmul_args:
        op = op + generate_random_matrix_batch(arg_name, arg_shape, arg_elemtype)
        if arg_name == "acc":
            op = op + generate_random_matrix_batch(
                "acc_copy", arg_shape, arg_elemtype, increment_seed=False
            )

    gen_names_and_types = lambda args_list: (
        ", ".join(["%" + name for name, shape, ty in args_list]),
        ", ".join(["!hal.buffer_view" for a in args_list]),
    )
    matmul_argnames, matmul_argtypes = gen_names_and_types(matmul_args)
    check_argnames, check_argtypes = gen_names_and_types(check_args)

    op += (
        f"  %result = util.call @module.{function.name}({matmul_argnames}) : ({matmul_argtypes}) -> !hal.buffer_view\n"
        f"  %batch = arith.constant {batch_shape.batch} : i64\n"
        f"  %m = arith.constant {batch_shape.m} : i64\n"
        f"  %k = arith.constant {batch_shape.k} : i64\n"
        f"  %n = arith.constant {batch_shape.n} : i64\n"
        f"  %transpose_rhs = arith.constant {transpose_rhs_val} : i32\n"
        f"  util.call @matmul_test.check_batch_matmul_results(%device, %batch, %m, %k, %n, %transpose_rhs, {check_argnames}, {'%acc_copy' if batch_shape.accumulate else '%acc'}, %result) : (!hal.device, i64, i64, i64, i64, i32, {check_argtypes}, !hal.buffer_view, !hal.buffer_view) -> ()\n"
        f"  util.return\n"
        f"}}\n"
    )

    return TestCall(function=function, op=op)


# Helper to generate random matrix for batch (3D tensor)
def generate_random_matrix_batch(
    name: str, matrix_shape: list, element_type: MatrixElemTypeId, increment_seed=True
):
    global random_matrix_seed
    if increment_seed:
        random_matrix_seed += 1
    # For batch matmul, we flatten batch*m or batch*n to 2D for generation
    batch, dim0, dim1 = matrix_shape
    flattened_dim0 = batch * dim0
    return (
        f"  %{name}_dim0 = arith.constant {flattened_dim0} : i64\n"
        f"  %{name}_dim1 = arith.constant {dim1} : i64\n"
        f"  %{name}_element_type = hal.element_type<{element_type.value}> : i32\n"
        f"  %{name}_seed = arith.constant {random_matrix_seed} : i32\n"
        f"  %{name}_2d = util.call @matmul_test.generate_random_matrix(%device, %{name}_dim0, %{name}_dim1, %{name}_element_type, %{name}_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view\n"
        f"  %{name}_batch_dim = arith.constant {batch} : i64\n"
        f"  %{name}_dim0_3d = arith.constant {dim0} : i64\n"
        f"  %{name}_dim1_3d = arith.constant {dim1} : i64\n"
        f"  %{name} = util.call @matmul_test.reshape_3d(%{name}_2d, %{name}_batch_dim, %{name}_dim0_3d, %{name}_dim1_3d) : (!hal.buffer_view, i64, i64, i64) -> !hal.buffer_view\n"
    )

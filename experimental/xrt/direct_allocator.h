// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_XRT_DIRECT_ALLOCATOR_H_
#define EXPERIMENTAL_XRT_DIRECT_ALLOCATOR_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

#include "xrt.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"
#include "xrt/xrt_bo.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Creates a XRT memory allocator.
iree_status_t iree_hal_xrt_allocator_create(
    xrt::device device,
    iree_allocator_t host_allocator, iree_hal_allocator_t** out_allocator);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // EXPERIMENTAL_XRT_DIRECT_ALLOCATOR_H_
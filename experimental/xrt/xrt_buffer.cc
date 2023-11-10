// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/xrt/xrt_buffer.h"

#include "iree/base/api.h"
#include "iree/base/target_platform.h"
#include "iree/base/tracing.h"
#include "iree/hal/api.h"

#include "xrt.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"
#include "xrt/xrt_bo.h"


typedef struct iree_hal_xrt_buffer_t {
  iree_hal_buffer_t base;
  xrt::bo buffer;
  iree_hal_buffer_release_callback_t release_callback;
} iree_hal_xrt_buffer_t;

namespace {
extern const iree_hal_buffer_vtable_t iree_hal_xrt_buffer_vtable;
}

static iree_hal_xrt_buffer_t* iree_hal_xrt_buffer_cast(iree_hal_buffer_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_xrt_buffer_vtable);
  return (iree_hal_xrt_buffer_t*)base_value;
}

static const iree_hal_xrt_buffer_t* iree_hal_xrt_buffer_const_cast(
    const iree_hal_buffer_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_xrt_buffer_vtable);
  return (const iree_hal_xrt_buffer_t*)base_value;
}

iree_status_t iree_hal_xrt_buffer_wrap(
    xrt::bo xrt_buffer, iree_hal_allocator_t* allocator, iree_hal_memory_type_t memory_type,
    iree_hal_memory_access_t allowed_access, iree_hal_buffer_usage_t allowed_usage,
    iree_device_size_t allocation_size, iree_device_size_t byte_offset,
    iree_device_size_t byte_length, iree_hal_buffer_release_callback_t release_callback,
    iree_hal_buffer_t** out_buffer) {
  IREE_ASSERT_ARGUMENT(allocator);
  IREE_ASSERT_ARGUMENT(out_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_allocator_t host_allocator = iree_hal_allocator_host_allocator(allocator);
  iree_hal_xrt_buffer_t* buffer = NULL;
  iree_status_t status = iree_allocator_malloc(host_allocator, sizeof(*buffer), (void**)&buffer);
  if (iree_status_is_ok(status)) {
    iree_hal_buffer_initialize(host_allocator, allocator, &buffer->base, allocation_size,
                               byte_offset, byte_length, memory_type, allowed_access, allowed_usage,
                               &iree_hal_xrt_buffer_vtable, &buffer->base);
    buffer->buffer = xrt_buffer;  // +1
    buffer->release_callback = release_callback;
    *out_buffer = &buffer->base;
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_xrt_buffer_destroy(iree_hal_buffer_t* base_buffer) {
  iree_hal_xrt_buffer_t* buffer = iree_hal_xrt_buffer_cast(base_buffer);
  iree_allocator_t host_allocator = base_buffer->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)iree_hal_buffer_allocation_size(base_buffer));

  if (buffer->release_callback.fn) {
    buffer->release_callback.fn(buffer->release_callback.user_data, base_buffer);
  }
  iree_allocator_free(host_allocator, buffer);

  IREE_TRACE_ZONE_END(z0);
}

xrt::bo iree_hal_xrt_buffer_handle(const iree_hal_buffer_t* base_buffer) {
  const iree_hal_xrt_buffer_t* buffer = iree_hal_xrt_buffer_const_cast(base_buffer);
  return buffer->buffer;
}

static iree_status_t iree_hal_xrt_buffer_invalidate_range(iree_hal_buffer_t* base_buffer,
                                                            iree_device_size_t local_byte_offset,
                                                            iree_device_size_t local_byte_length) {
  return iree_ok_status();
}

static iree_status_t iree_hal_xrt_buffer_flush_range(iree_hal_buffer_t* base_buffer,
                                                       iree_device_size_t local_byte_offset,
                                                       iree_device_size_t local_byte_length) {
  return iree_ok_status();
}


static iree_status_t iree_hal_xrt_buffer_map_range(iree_hal_buffer_t* base_buffer,
                                                     iree_hal_mapping_mode_t mapping_mode,
                                                     iree_hal_memory_access_t memory_access,
                                                     iree_device_size_t local_byte_offset,
                                                     iree_device_size_t local_byte_length,
                                                     iree_hal_buffer_mapping_t* mapping) {
  iree_hal_xrt_buffer_t* buffer = iree_hal_xrt_buffer_cast(base_buffer);

  // TODO(benvanik): add upload/download for unmapped buffers.
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_memory_type(
      iree_hal_buffer_memory_type(base_buffer), IREE_HAL_MEMORY_TYPE_HOST_VISIBLE));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_usage(iree_hal_buffer_allowed_usage(base_buffer),
                                                      IREE_HAL_BUFFER_USAGE_MAPPING));

  void* host_ptr = buffer->buffer.map();
  IREE_ASSERT(host_ptr != NULL);  // Should be guaranteed by previous checks.
  uint8_t* data_ptr = (uint8_t*)host_ptr + local_byte_offset;
  // If we mapped for discard scribble over the bytes. This is not a mandated behavior but it will
  // make debugging issues easier. Alternatively for heap buffers we could reallocate them such that
  // ASAN yells, but that would only work if the entire buffer was discarded.
#ifndef NDEBUG
  if (iree_any_bit_set(memory_access, IREE_HAL_MEMORY_ACCESS_DISCARD)) {
    memset(data_ptr, 0xCD, local_byte_length);
  }
#endif  // !NDEBUG

  iree_status_t status = iree_ok_status();
  mapping->contents = iree_make_byte_span(data_ptr, local_byte_length);
  return status;
}

static iree_status_t iree_hal_xrt_buffer_unmap_range(iree_hal_buffer_t* base_buffer,
                                                       iree_device_size_t local_byte_offset,
                                                       iree_device_size_t local_byte_length,
                                                       iree_hal_buffer_mapping_t* mapping) {
  return iree_ok_status();
}

namespace {
const iree_hal_buffer_vtable_t iree_hal_xrt_buffer_vtable = {
    /*.recycle = */iree_hal_buffer_recycle,
    /*.destroy = */iree_hal_xrt_buffer_destroy,
    /*.map_range = */iree_hal_xrt_buffer_map_range,
    /*.unmap_range = */iree_hal_xrt_buffer_unmap_range,
    /*.invalidate_range = */iree_hal_xrt_buffer_invalidate_range,
    /*.flush_range = */iree_hal_xrt_buffer_flush_range,
};
} // namespace
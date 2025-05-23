//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___CXX03___RANDOM_GENERATE_CANONICAL_H
#define _LIBCPP___CXX03___RANDOM_GENERATE_CANONICAL_H

#include <__cxx03/__config>
#include <__cxx03/__random/log2.h>
#include <__cxx03/cstdint>
#include <__cxx03/limits>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__cxx03/__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

// generate_canonical

template <class _RealType, size_t __bits, class _URNG>
_LIBCPP_HIDE_FROM_ABI _RealType generate_canonical(_URNG& __g) {
  const size_t __dt    = numeric_limits<_RealType>::digits;
  const size_t __b     = __dt < __bits ? __dt : __bits;
  const size_t __log_r = __log2<uint64_t, _URNG::_Max - _URNG::_Min + uint64_t(1)>::value;
  const size_t __k     = __b / __log_r + (__b % __log_r != 0) + (__b == 0);
  const _RealType __rp = static_cast<_RealType>(_URNG::max() - _URNG::min()) + _RealType(1);
  _RealType __base     = __rp;
  _RealType __sp       = __g() - _URNG::min();
  for (size_t __i = 1; __i < __k; ++__i, __base *= __rp)
    __sp += (__g() - _URNG::min()) * __base;
  return __sp / __base;
}

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP___CXX03___RANDOM_GENERATE_CANONICAL_H

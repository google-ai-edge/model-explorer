// Copyright 2024 The AI Edge Model Explorer Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#include "tools/attribute_printer.h"

#include <algorithm>
#include <cstdint>

#include "absl/log/log.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "tools/shardy_utils.h"

namespace tooling {
namespace visualization_client {
namespace {

// Checks whether character c is printable. This method is borrowed from
// third_party/llvm/llvm-project/mlir/lib/IR/AsmPrinter.cpp.
inline bool IsPrint(char c) {
  unsigned char uc = static_cast<unsigned char>(c);
  return (0x20 <= uc) && (uc <= 0x7E);
}

// This method is majorly borrowed from
// third_party/llvm/llvm-project/mlir/lib/IR/AsmPrinter.cpp.
void PrintDenseElementsAttrImpl(bool is_splat, const mlir::ShapedType& type,
                                llvm::function_ref<void(unsigned)> PrintEltFn,
                                const int64_t size_limit,
                                llvm::raw_ostream& os) {
  // Compares the size of the tensor to the size limit and adopts the smaller
  // one. Sets size limit to -1 to print all elements.
  const unsigned num_elements =
      (size_limit < 0) ? type.getNumElements()
                       : std::min(type.getNumElements(), size_limit);
  if (num_elements == 0) return;

  // Special case for 0-d and splat tensors.
  if (is_splat) return PrintEltFn(0);

  // We use a mixed-radix counter to iterate through the shape. When we bump a
  // non-least-significant digit, we emit a close bracket. When we next emit an
  // element we re-open all closed brackets.

  // The mixed-radix counter, with radices in 'shape'.
  int64_t rank = type.getRank();
  llvm::SmallVector<unsigned, 4> counter(rank, 0);
  // The number of brackets that have been opened and not closed.
  unsigned open_brackets = 0;

  llvm::ArrayRef<int64_t> shape = type.getShape();
  auto BumpCounter = [&] {
    // Bump the least significant digit.
    ++counter[rank - 1];
    // Iterate backwards bubbling back the increment.
    for (unsigned i = rank - 1; i > 0; --i)
      if (counter[i] >= shape[i]) {
        // Index 'i' is rolled over. Bump (i-1) and close a bracket.
        counter[i] = 0;
        ++counter[i - 1];
        --open_brackets;
        os << ']';
      }
  };

  for (unsigned idx = 0, e = num_elements; idx != e; ++idx) {
    if (idx != 0) os << ", ";
    while (open_brackets++ < rank) os << '[';
    open_brackets = rank;
    PrintEltFn(idx);
    BumpCounter();
  }
  while (open_brackets-- > 0) os << ']';
}

}  // namespace

void PrintString(llvm::StringRef str, llvm::raw_ostream& os) {
  for (unsigned char c : str) {
    if (IsPrint(c)) {
      os << c;
    }
  }
}

void PrintIntValue(const llvm::APInt& value, const mlir::Type& type,
                   llvm::raw_ostream& os) {
  if (type.isInteger(1)) {
    os << (value.getBoolValue() ? "true" : "false");
  } else {
    value.print(os, !type.isUnsignedInteger());
  }
}

void PrintFloatValue(const llvm::APFloat& ap_value, llvm::raw_ostream& os) {
  // We would like to output the FP constant value in exponential notation,
  // but we cannot do this if doing so will lose precision.  Check here to
  // make sure that we only output it in exponential format if we can parse
  // the value back and get the same value.
  llvm::SmallString<128> str_value;
  ap_value.toString(str_value, /*FormatPrecision=*/6, /*FormatMaxPadding=*/0,
                    /*TruncateZero=*/false);

  // Parse back the stringized version and check that the value is equal
  // (i.e., there is no precision loss).
  if (llvm::APFloat(ap_value.getSemantics(), str_value)
          .bitwiseIsEqual(ap_value)) {
    os << str_value;
    return;
  }

  // If it is not, use the default format of APFloat instead of the
  // exponential notation.
  str_value.clear();
  ap_value.toString(str_value);
  os << str_value;
}

void PrintDenseStringElementsAttr(const mlir::DenseStringElementsAttr& attr,
                                  const int64_t size_limit,
                                  llvm::raw_ostream& os) {
  llvm::ArrayRef<llvm::StringRef> data = attr.getRawStringData();
  auto PrintFn = [&](unsigned index) { PrintString(data[index], os); };
  PrintDenseElementsAttrImpl(attr.isSplat(), attr.getType(), PrintFn,
                             size_limit, os);
}

void PrintDenseIntOrFPElementsAttr(const mlir::DenseIntOrFPElementsAttr& attr,
                                   const int64_t size_limit,
                                   llvm::raw_ostream& os) {
  mlir::ShapedType type = attr.getType();
  mlir::Type element_type = type.getElementType();

  if (element_type.isIntOrIndex()) {
    auto value_it = attr.value_begin<llvm::APInt>();
    auto PrintFn = [&](unsigned index) {
      PrintIntValue(*(value_it + index), element_type, os);
    };
    PrintDenseElementsAttrImpl(attr.isSplat(), type, PrintFn, size_limit, os);
  } else {
    if (!llvm::isa<mlir::FloatType>(element_type)) {
      LOG(ERROR) << "unexpected element type";
      return;
    }
    auto value_it = attr.value_begin<llvm::APFloat>();
    auto PrintFn = [&](unsigned index) {
      PrintFloatValue(*(value_it + index), os);
    };
    PrintDenseElementsAttrImpl(attr.isSplat(), type, PrintFn, size_limit, os);
  }
}

void PrintDenseElementsAttr(const mlir::DenseElementsAttr& attr,
                            const int64_t size_limit, llvm::raw_ostream& os) {
  if (auto string_attr = llvm::dyn_cast<mlir::DenseStringElementsAttr>(attr);
      string_attr != nullptr) {
    return PrintDenseStringElementsAttr(string_attr, size_limit, os);
  }

  PrintDenseIntOrFPElementsAttr(
      llvm::cast<mlir::DenseIntOrFPElementsAttr>(attr), size_limit, os);
}

void PrintAttribute(const mlir::Attribute& attr, const int64_t size_limit,
                    llvm::raw_string_ostream& os) {
  if (const auto& elm_attr =
          llvm::dyn_cast_or_null<mlir::DenseElementsAttr>(attr)) {
    PrintDenseElementsAttr(elm_attr, size_limit, os);
  } else if (const auto& str_attr =
                 llvm::dyn_cast_or_null<mlir::StringAttr>(attr)) {
    PrintString(str_attr.getValue(), os);
  } else if (IsShardyDialect(attr)) {
    PrintShardyAttribute(attr, os);
  } else {
    attr.print(os, /*elideType=*/true);
  }
}

}  // namespace visualization_client
}  // namespace tooling

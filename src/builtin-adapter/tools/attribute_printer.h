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

#ifndef TOOLS_ATTRIBUTE_PRINTER_H_
#define TOOLS_ATTRIBUTE_PRINTER_H_

#include <cstdint>

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"

namespace tooling {
namespace visualization_client {

// Prints the string. Escape chars and non-ASCII chars are omitted.
void PrintString(llvm::StringRef str, llvm::raw_ostream& os);

// Prints an integer value. The integer with width 1 is printed as bool string.
void PrintIntValue(const llvm::APInt& value, const mlir::Type& type,
                   llvm::raw_ostream& os);

// Prints a floating point value in a way that the parser will be able to
// round-trip losslessly.
void PrintFloatValue(const llvm::APFloat& ap_value, llvm::raw_ostream& os);

// Prints the string elements of a DenseStringElementsAttr. Uses size_limit to
// control the number of values to be printed, set to -1 to print all.
void PrintDenseStringElementsAttr(const mlir::DenseStringElementsAttr& attr,
                                  int64_t size_limit, llvm::raw_ostream& os);

// Prints the DenseIntOrFPElementsAttr. Uses size_limit to control the number
// of values to be printed, set to -1 to print all.
void PrintDenseIntOrFPElementsAttr(const mlir::DenseIntOrFPElementsAttr& attr,
                                   int64_t size_limit,
                                   llvm::raw_string_ostream& os);

// Prints the DenseElementsAttr. Uses size_limit to control the number of
// values to be printed, set to -1 to print all.
void PrintDenseElementsAttr(const mlir::DenseElementsAttr& attr,
                            int64_t size_limit, llvm::raw_ostream& os);

// Prints the attribute value to stream. Uses size_limit to control the number
// of values to be printed (only affect DenseElements), set to -1 to print all.
void PrintAttribute(const mlir::Attribute& attr, int64_t size_limit,
                    llvm::raw_string_ostream& os);

}  // namespace visualization_client
}  // namespace tooling

#endif  // TOOLS_ATTRIBUTE_PRINTER_H_

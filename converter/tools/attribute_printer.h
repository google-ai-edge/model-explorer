#ifndef TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_GOOGLE_TOOLING_TOOLS_ATTRIBUTE_PRINTER_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_GOOGLE_TOOLING_TOOLS_ATTRIBUTE_PRINTER_H_

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

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_GOOGLE_TOOLING_TOOLS_ATTRIBUTE_PRINTER_H_

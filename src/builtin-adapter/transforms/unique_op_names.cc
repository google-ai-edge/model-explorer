// Copyright 2025 The AI Edge Model Explorer Authors.
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

#include <memory>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "transforms/passes.h"

#define GEN_PASS_DEF_UNIQUEOPNAMES
#include "transforms/passes.h.inc"

namespace tooling {
namespace visualization_client {
namespace {

// This pass ensures that every operation with a NameLoc has a unique name
// across the entire module, which is critical for graph visualization tools
// that rely on names as node identifiers.
//
// Key features:
// 1.  Uniqueness: Guarantees unique names for all ops with NameLocs.
// 2.  Hierarchical Naming for Scoping Ops: Ops that have regions (eg.
//     control flow, functions) and are nested within other ops with NameLocs
//     will have a hierarchical name like "parent_name/op_name". This helps
//     preserve the structural context in the visualization.
// 3.  Location Inheritance for Leaf Ops: Nested ops *without* their own
//     NameLoc are grouped with their parent by inheriting their parent's
//     location. This applies regardless of whether they have regions.
struct UniqueOpNamesPass : public impl::UniqueOpNamesBase<UniqueOpNamesPass> {
 public:
  void runOnOperation() override {
    llvm::StringSet<> existing_names;
    unsigned uniquing_counter = 0;

    // Starts the recursion from the top-level module op.
    UniquifyOpAndChildren(getOperation(), mlir::UnknownLoc::get(&getContext()),
                          existing_names, uniquing_counter);
  }

 private:
  // Builds the base name for an op, potentially prefixing with the parent's
  // name if nested.
  llvm::SmallString<256> BuildBaseName(const bool is_nested,
                                       mlir::NameLoc name_loc,
                                       mlir::Location inherited_loc) {
    llvm::SmallString<256> out_name;
    llvm::StringRef original_name = name_loc.getName().getValue();

    if (is_nested) {
      if (auto inherited = mlir::dyn_cast<mlir::NameLoc>(inherited_loc)) {
        // Nested op: Build hierarchical name "parent_name/op_name".
        out_name.append(inherited.getName().getValue());
        out_name.append("/");
        // Appends the most specific part of the original name (in case
        // original_name itself was already hierarchical).
        llvm::SmallVector<llvm::StringRef, 4> parts;
        original_name.split(parts, '/');
        out_name.append(parts.back());
      } else {
        // Nested, but parent had no NameLoc. Fall back to original.
        out_name.append(original_name);
      }
    } else {
      // Top-level op. Use its name directly.
      out_name.append(original_name);
    }
    return out_name;
  }

  // Recursively traverses the operations, uniquifying their names and
  // propagating locations.
  //
  // - op: The current operation to process.
  // - inherited_loc: The location from the parent scope.
  // - existing_names: Set of names used so far, for uniquing.
  // - uniquing_counter: Counter for generating unique name suffixes.
  void UniquifyOpAndChildren(mlir::Operation* op, mlir::Location inherited_loc,
                             llvm::StringSet<>& existing_names,
                             unsigned& uniquing_counter) {
    mlir::Operation* parent_op = op->getParentOp();
    const bool is_nested =
        parent_op && !llvm::isa<mlir::func::FuncOp, mlir::ModuleOp>(parent_op);

    mlir::Location current_loc = op->getLoc();
    mlir::Location final_loc = current_loc;

    if (auto name_loc = mlir::dyn_cast<mlir::NameLoc>(current_loc)) {
      // This op has a NameLoc, potentially update it for hierarchy and
      // uniqueness.
      llvm::SmallString<256> base_name =
          BuildBaseName(is_nested, name_loc, inherited_loc);

      llvm::SmallString<256> unique_name = base_name;
      if (existing_names.count(base_name)) {
        unique_name = mlir::SymbolTable::generateSymbolName<256>(
            base_name,
            [&](llvm::StringRef c) { return existing_names.count(c); },
            uniquing_counter);
      }

      existing_names.insert(unique_name);
      if (name_loc.getName().getValue() != unique_name) {
        final_loc = mlir::NameLoc::get(
            mlir::StringAttr::get(&getContext(), unique_name),
            name_loc.getChildLoc());
        op->setLoc(final_loc);
      } else {
        final_loc = current_loc;
      }
    } else if (is_nested) {
      // Nested op without a NameLoc: Inherit the parent's location.
      final_loc = inherited_loc;
      op->setLoc(final_loc);
    } else {
      // Top-level op without NameLoc: Keep its location (eg. UnknownLoc).
      final_loc = current_loc;
    }

    // Determines the location to pass down to children.
    // Children inherit the final location of the current op, but only if the
    // current op defines a new scope (has regions).
    const bool op_has_regions = llvm::any_of(
        op->getRegions(), [](mlir::Region& region) { return !region.empty(); });
    mlir::Location loc_for_children =
        op_has_regions ? final_loc : inherited_loc;

    // Recurses into any regions this op may have.
    for (mlir::Region& region : op->getRegions()) {
      for (mlir::Block& block : region) {
        for (mlir::Operation& child_op : block) {
          UniquifyOpAndChildren(&child_op, loc_for_children, existing_names,
                                uniquing_counter);
        }
      }
    }
  }
};
}  // namespace

std::unique_ptr<mlir::Pass> CreateUniqueOpNamesPass() {
  return std::make_unique<UniqueOpNamesPass>();
}

}  // namespace visualization_client
}  // namespace tooling

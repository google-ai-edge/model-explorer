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

#include "llvm/ADT/SmallString.h"
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

// This pass ensures that every operation with a NameLoc has a unique name,
// which is critical for graph visualization tools that rely on names as node
// identifiers.
//
// A key feature of this pass is its handling of nested operations (ops inside
// regions). To maintain the visual hierarchy in the graph (e.g., showing an
// 'add' op inside a 'reduce_window' op), nested ops have their locations set to
// match their parent's location. This makes them appear as part of the same
// logical group in the visualizer.
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
  void UniquifyOpAndChildren(mlir::Operation* op, mlir::Location inherited_loc,
                             llvm::StringSet<>& existing_names,
                             unsigned& uniquing_counter) {
    mlir::Operation* parent_op = op->getParentOp();
    bool is_nested =
        parent_op && !llvm::isa<mlir::func::FuncOp, mlir::ModuleOp>(parent_op);

    mlir::Location loc_for_children = mlir::UnknownLoc::get(&getContext());

    if (is_nested) {
      op->setLoc(inherited_loc);
      loc_for_children = inherited_loc;
    } else {
      mlir::Location loc_to_process = op->getLoc();
      if (auto name_loc = llvm::dyn_cast<mlir::NameLoc>(loc_to_process)) {
        llvm::StringRef current_name = name_loc.getName().getValue();
        if (!existing_names.insert(current_name).second) {
          // Name collision detected, generates a new unique name.
          llvm::SmallString<128> new_name =
              mlir::SymbolTable::generateSymbolName<128>(
                  current_name,
                  [&](llvm::StringRef c) { return existing_names.count(c); },
                  uniquing_counter);
          op->setLoc(
              mlir::NameLoc::get(mlir::StringAttr::get(&getContext(), new_name),
                                 name_loc.getChildLoc()));
          existing_names.insert(new_name);
        }
      }
      // This op's own location is the one that its direct children will
      // inherit.
      loc_for_children = op->getLoc();
    }

    // Recurses into any regions this op may have, passing down the determined
    // location for its children to inherit.
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

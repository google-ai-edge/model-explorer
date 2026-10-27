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

#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "adapters/mlir/translations.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/dialect/VhloOps.h"
#include "stablehlo/tests/CheckOps.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"

static mlir::TranslateFromMLIRRegistration
    JaxConvertedMlirToJsonTranslateRegistration(
        "mlir-to-json", "Translates an MLIR dump to a JSON graph.",
        model_explorer::adapter::MlirToJsonTranslate,
        [](mlir::DialectRegistry& registry) {
          registry.insert<
              mlir::TFL::TensorFlowLiteDialect, mlir::TF::TensorFlowDialect,
              mlir::stablehlo::StablehloDialect, mlir::chlo::ChloDialect,
              mlir::vhlo::VhloDialect, mlir::func::FuncDialect,
              mlir::arith::ArithDialect, mlir::shape::ShapeDialect,
              mlir::scf::SCFDialect, mlir::stablehlo::check::CheckDialect,
              mlir::sdy::SdyDialect>();
        });

int main(int argc, char** argv) {
  return failed(mlir::mlirTranslateMain(argc, argv, ""));
}

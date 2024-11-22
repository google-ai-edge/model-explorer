# Copyright 2024 The AI Edge Model Explorer Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import logging
import model_explorer
import re
import tempfile
import time
import torch
import torchvision
from playwright.sync_api import Page, expect
from PIL import Image, ImageChops
from pathlib import Path

LOCAL_SERVER = "http://127.0.0.1:8080//"
ROOT_DIR = Path(__file__).parent.parent
TEST_FILES_DIR = ROOT_DIR / "test/test_models"
TMP_SCREENSHOT_DIR = ROOT_DIR / "build"
DEBUG_SCREENSHOT_DIR = TMP_SCREENSHOT_DIR / "debug"
EXPECTED_SCREENSHOT_DIR = ROOT_DIR / "test/screenshots_golden/chrome-linux"


# Compares two images, the similarity is determined by calculating the percentage
# of mismatched pixels. `threshold` can be set between 0 to 1, 0 means the images
# are identical. Default to 40 pixel tolerance 40/(1280*720) = 0.000043.
def matched_images(
    actual_image_path: Path,
    expected_image_path: Path,
    threshold: float = 0.000043,
):
  actual_image = Image.open(actual_image_path).convert("L")
  expected_image = Image.open(expected_image_path).convert("L")

  if actual_image.size != expected_image.size:
    return False
  diff = ImageChops.difference(actual_image, expected_image)
  diff_list = list(diff.getdata())
  mismatch_ratio = sum(pixel != 0 for pixel in diff_list) * 1.0 / len(diff_list)
  if mismatch_ratio > threshold:
    DEBUG_SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    diff.save(DEBUG_SCREENSHOT_DIR / actual_image_path.name)
    logging.error(
        "".join([
            "Screenshot [",
            actual_image_path.name,
            "] has mismatch ratio of [",
            str(mismatch_ratio),
            "].",
        ])
    )
    return False
  return True


def delay_view_model(page: Page):
  page.get_by_role("button", name="View selected models").click()
  time.sleep(2)  # Delay for the animation


def delay_click_canvas(page: Page, x: int, y: int):
  time.sleep(2)  # Delay for the animation
  page.locator("canvas").first.click(position={"x": x, "y": y})


def delay_take_screenshot(page: Page, file_path: str):
  time.sleep(2)  # Delay for the animation
  page.screenshot(path=file_path)


def take_and_compare_screenshot(page: Page, name: str):
  actual_image_path = TMP_SCREENSHOT_DIR / name
  delay_take_screenshot(page, actual_image_path)
  expected_image_path = EXPECTED_SCREENSHOT_DIR / name
  assert matched_images(actual_image_path, expected_image_path)


def test_homepage(page: Page):
  page.goto(LOCAL_SERVER)
  expect(page).to_have_title(re.compile("Model Explorer"))
  take_and_compare_screenshot(page, "homepage.png")


def test_litert_direct_adapter(page: Page):
  page.goto(LOCAL_SERVER)
  page.get_by_placeholder("Absolute file paths (").fill(
      TEST_FILES_DIR / "fully_connected.tflite"
  )
  page.get_by_role("button", name="Add").click()
  page.get_by_text("arrow_drop_down").click()
  page.get_by_text("TFLite adapter (Flatbuffer)").click()
  delay_view_model(page)
  page.locator("canvas").first.click(position={"x": 469, "y": 340})

  take_and_compare_screenshot(page, "litert_direct.png")


def test_litert_mlir_adapter(page: Page):
  page.goto(LOCAL_SERVER)
  page.get_by_placeholder("Absolute file paths (").fill(
      TEST_FILES_DIR / "fully_connected.tflite"
  )
  page.get_by_role("button", name="Add").click()
  page.get_by_text("arrow_drop_down").click()
  page.get_by_text("TFLite adapter (MLIR)").click()
  delay_view_model(page)
  page.locator("canvas").first.click(position={"x": 514, "y": 332})

  take_and_compare_screenshot(page, "litert_mlir.png")


def test_tf_mlir_adapter(page: Page):
  page.goto(LOCAL_SERVER)
  page.get_by_placeholder("Absolute file paths (").fill(
      TEST_FILES_DIR / "simple_add/saved_model.pb"
  )
  page.get_by_role("button", name="Add").click()
  page.get_by_text("arrow_drop_down").click()
  page.get_by_text("TF adapter (MLIR) Default").click()
  delay_view_model(page)
  page.locator("canvas").first.click(position={"x": 444, "y": 281})

  take_and_compare_screenshot(page, "tf_mlir.png")


def test_tf_direct_adapter(page: Page):
  page.goto(LOCAL_SERVER)
  page.get_by_placeholder("Absolute file paths (").fill(
      TEST_FILES_DIR / "simple_add/saved_model.pb"
  )
  page.get_by_role("button", name="Add").click()
  page.get_by_text("arrow_drop_down").click()
  page.get_by_text("TF adapter (direct)").click()
  delay_view_model(page)
  page.get_by_text("__inference__traced_save_36", exact=True).click()
  page.get_by_text("__inference_add_6").click()
  delay_click_canvas(page, 205, 265)

  take_and_compare_screenshot(page, "tf_direct.png")


def test_tf_graphdef_adapter(page: Page):
  page.goto(LOCAL_SERVER)
  page.get_by_placeholder("Absolute file paths (").fill(
      TEST_FILES_DIR / "graphdef_foo.pbtxt"
  )
  page.get_by_role("button", name="Add").click()
  delay_view_model(page)
  page.locator("canvas").first.click(position={"x": 468, "y": 344})

  take_and_compare_screenshot(page, "graphdef.png")


def test_shlo_mlir_adapter(page: Page):
  page.goto(LOCAL_SERVER)
  page.get_by_placeholder("Absolute file paths (").fill(
      TEST_FILES_DIR / "stablehlo_sin.mlir"
  )
  page.get_by_role("button", name="Add").click()
  delay_view_model(page)
  page.get_by_text("unfold_more_double").click()
  delay_click_canvas(page, 454, 416)

  take_and_compare_screenshot(page, "shlo_mlir.png")


def test_pytorch(page: Page):
  # Serialize a pytorch model.
  model = torchvision.models.mobilenet_v2().eval()
  inputs = (torch.rand([1, 3, 224, 224]),)
  ep = torch.export.export(model, inputs)
  tmp_dir = tempfile.gettempdir()
  pt2_file_path = f"{tmp_dir}/pytorch.pt2"
  torch.export.save(ep, pt2_file_path)

  # Load into ME.
  page.goto(LOCAL_SERVER)
  page.get_by_placeholder("Absolute file paths (").fill(pt2_file_path)
  page.get_by_role("button", name="Add").click()
  delay_view_model(page)
  page.locator("canvas").first.click(position={"x": 458, "y": 334})

  take_and_compare_screenshot(page, "pytorch.png")


def test_reuse_server_non_pytorch(page: Page):
  # Load a tflite model
  page.goto(LOCAL_SERVER)
  page.get_by_placeholder("Absolute file paths (").fill(
      TEST_FILES_DIR / "fully_connected.tflite"
  )
  page.get_by_role("button", name="Add").click()
  delay_view_model(page)

  # Load a mlir graph and reuse the existing server.
  mlir_model_path = TEST_FILES_DIR / "stablehlo_sin.mlir"
  model_explorer.visualize(
      model_paths=mlir_model_path.as_posix(), reuse_server=True
  )
  time.sleep(2)  # Delay for the animation

  take_and_compare_screenshot(page, "reuse_server_non_pytorch.png")


def test_reuse_server_pytorch(page: Page):
  # Load a tflite model
  page.goto(LOCAL_SERVER)
  page.get_by_placeholder("Absolute file paths (").fill(
      TEST_FILES_DIR / "fully_connected.tflite"
  )
  page.get_by_role("button", name="Add").click()
  delay_view_model(page)

  # Load a pytorch model and reuse the existing server.
  model = torchvision.models.mobilenet_v2().eval()
  inputs = (torch.rand([1, 3, 224, 224]),)
  ep = torch.export.export(model, inputs)
  model_explorer.visualize_pytorch(
      name="test pytorch", exported_program=ep, reuse_server=True
  )
  time.sleep(2)  # Delay for the animation

  take_and_compare_screenshot(page, "reuse_server_pytorch.png")


def test_reuse_server_pytorch_from_config(page: Page):
  # Load a tflite model
  page.goto(LOCAL_SERVER)
  page.get_by_placeholder("Absolute file paths (").fill(
      TEST_FILES_DIR / "fully_connected.tflite"
  )
  page.get_by_role("button", name="Add").click()
  delay_view_model(page)

  # Load a pytorch model and reuse the existing server through config.
  model = torchvision.models.mobilenet_v2().eval()
  inputs = (torch.rand([1, 3, 224, 224]),)
  ep = torch.export.export(model, inputs)
  config = model_explorer.config()
  config.add_model_from_pytorch("test pytorch", ep).set_reuse_server()
  model_explorer.visualize_from_config(config)
  time.sleep(2)  # Delay for the animation

  take_and_compare_screenshot(page, "reuse_server_pytorch_from_config.png")


def test_reuse_server_two_pytorch_models(page: Page):
  # Serialize a pytorch model (mobilenet v2).
  model = torchvision.models.mobilenet_v2().eval()
  inputs = (torch.rand([1, 3, 224, 224]),)
  ep = torch.export.export(model, inputs)
  pt2_file_path = tempfile.NamedTemporaryFile(suffix=".pt2")
  torch.export.save(ep, pt2_file_path.name)

  # Load it into ME.
  page.goto(LOCAL_SERVER)
  page.get_by_placeholder("Absolute file paths (").fill(pt2_file_path.name)
  page.get_by_role("button", name="Add").click()
  delay_view_model(page)

  # Load another pytorch model (mobilenet v3) and reuse the existing server.
  model2 = torchvision.models.mobilenet_v3_small().eval()
  inputs2 = (torch.rand([1, 3, 224, 224]),)
  ep2 = torch.export.export(model2, inputs2)
  model_explorer.visualize_pytorch(
      name="test pytorch", exported_program=ep2, reuse_server=True
  )
  time.sleep(2)  # Delay for the animation

  # The screenshot should show "V3" in the center node.
  take_and_compare_screenshot(page, "reuse_server_two_pytorch_models.png")

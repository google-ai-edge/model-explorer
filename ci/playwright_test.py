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
import re
import time
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


def test_homepage(page: Page):
  page.goto(LOCAL_SERVER)
  expect(page).to_have_title(re.compile("Model Explorer"))
  actual_image_path = TMP_SCREENSHOT_DIR / "homepage.png"
  delay_take_screenshot(page, actual_image_path)
  expected_image_path = EXPECTED_SCREENSHOT_DIR / "homepage.png"
  assert matched_images(actual_image_path, expected_image_path)


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
  actual_image_path = TMP_SCREENSHOT_DIR / "litert_direct.png"
  delay_take_screenshot(page, actual_image_path)
  expected_image_path = EXPECTED_SCREENSHOT_DIR / "litert_direct.png"
  assert matched_images(actual_image_path, expected_image_path)


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
  actual_image_path = TMP_SCREENSHOT_DIR / "litert_mlir.png"
  delay_take_screenshot(page, actual_image_path)
  expected_image_path = EXPECTED_SCREENSHOT_DIR / "litert_mlir.png"
  assert matched_images(actual_image_path, expected_image_path)


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
  actual_image_path = TMP_SCREENSHOT_DIR / "tf_mlir.png"
  delay_take_screenshot(page, actual_image_path)
  expected_image_path = EXPECTED_SCREENSHOT_DIR / "tf_mlir.png"
  assert matched_images(actual_image_path, expected_image_path)


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
  actual_image_path = TMP_SCREENSHOT_DIR / "tf_direct.png"
  delay_take_screenshot(page, actual_image_path)
  expected_image_path = EXPECTED_SCREENSHOT_DIR / "tf_direct.png"
  assert matched_images(actual_image_path, expected_image_path)


def test_tf_graphdef_adapter(page: Page):
  page.goto(LOCAL_SERVER)
  page.get_by_placeholder("Absolute file paths (").fill(
      TEST_FILES_DIR / "graphdef_foo.pbtxt"
  )
  page.get_by_role("button", name="Add").click()
  delay_view_model(page)
  page.locator("canvas").first.click(position={"x": 468, "y": 344})
  actual_image_path = TMP_SCREENSHOT_DIR / "graphdef.png"
  delay_take_screenshot(page, actual_image_path)
  expected_image_path = EXPECTED_SCREENSHOT_DIR / "graphdef.png"
  assert matched_images(actual_image_path, expected_image_path)


def test_shlo_mlir_adapter(page: Page):
  page.goto(LOCAL_SERVER)
  page.get_by_placeholder("Absolute file paths (").fill(
      TEST_FILES_DIR / "stablehlo_sin.mlir"
  )
  page.get_by_role("button", name="Add").click()
  delay_view_model(page)
  page.get_by_text("unfold_more_double").click()
  delay_click_canvas(page, 454, 416)
  actual_image_path = TMP_SCREENSHOT_DIR / "shlo_mlir.png"
  delay_take_screenshot(page, actual_image_path)
  expected_image_path = EXPECTED_SCREENSHOT_DIR / "shlo_mlir.png"
  assert matched_images(actual_image_path, expected_image_path)

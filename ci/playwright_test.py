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

import re
import time
from playwright.sync_api import Page, expect
import cv2
from pathlib import Path

LOCAL_SERVER = "http://127.0.0.1:8080//"
ROOT_DIR = Path(__file__).parent.parent
TEST_FILES_DIR = ROOT_DIR / "test/test_models"
TMP_SCREENSHOT_DIR = ROOT_DIR / "build"
EXPECTED_SCREENSHOT_DIR = ROOT_DIR / "test/screenshots_golden/chrome-linux"


def matched_images(
    actual_image_path, expected_image_path, threshold_ratio: int = 0.4
):
  actual_image = cv2.imread(actual_image_path)
  expected_image = cv2.imread(expected_image_path)

  orb = cv2.ORB_create()
  actual_keypoints, actual_descriptors = orb.detectAndCompute(
      actual_image, None
  )
  expected_keypoints, expected_descriptors = orb.detectAndCompute(
      expected_image, None
  )

  bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
  matches = bf.match(actual_descriptors, expected_descriptors)
  num_matches = len(matches)
  threshold = threshold_ratio * min(
      len(actual_keypoints), len(expected_keypoints)
  )

  if num_matches >= threshold:
    return True
  return False


def test_homepage(page: Page):
  page.goto(LOCAL_SERVER)
  expect(page).to_have_title(re.compile("Model Explorer"))
  actual_image_path = TMP_SCREENSHOT_DIR / "homepage.png"
  page.screenshot(path=actual_image_path)
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
  page.get_by_role("button", name="View selected models").click()
  page.locator("canvas").first.click(position={"x": 469, "y": 340})
  actual_image_path = TMP_SCREENSHOT_DIR / "litert_direct.png"
  page.screenshot(path=actual_image_path)
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
  page.get_by_role("button", name="View selected models").click()
  page.locator("canvas").first.click(position={"x": 514, "y": 332})
  actual_image_path = TMP_SCREENSHOT_DIR / "litert_mlir.png"
  page.screenshot(path=actual_image_path)
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
  page.get_by_role("button", name="View selected models").click()
  page.locator("canvas").first.click(position={"x": 444, "y": 281})
  actual_image_path = TMP_SCREENSHOT_DIR / "tf_mlir.png"
  page.screenshot(path=actual_image_path)
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
  page.get_by_role("button", name="View selected models").click()
  page.get_by_text("__inference__traced_save_36", exact=True).click()
  page.get_by_text("__inference_add_6").click()
  page.locator("canvas").first.click(position={"x": 723, "y": 278})
  actual_image_path = TMP_SCREENSHOT_DIR / "tf_direct.png"
  page.screenshot(path=actual_image_path)
  expected_image_path = EXPECTED_SCREENSHOT_DIR / "tf_direct.png"
  assert matched_images(actual_image_path, expected_image_path)


def test_tf_graphdef_adapter(page: Page):
  page.goto(LOCAL_SERVER)
  page.get_by_placeholder("Absolute file paths (").fill(
      TEST_FILES_DIR / "graphdef_foo.pbtxt"
  )
  page.get_by_role("button", name="Add").click()
  page.get_by_role("button", name="View selected models").click()
  page.locator("canvas").first.click(position={"x": 468, "y": 344})
  actual_image_path = TMP_SCREENSHOT_DIR / "graphdef.png"
  page.screenshot(path=actual_image_path)
  expected_image_path = EXPECTED_SCREENSHOT_DIR / "graphdef.png"
  assert matched_images(actual_image_path, expected_image_path)


def test_shlo_mlir_adapter(page: Page):
  page.goto(LOCAL_SERVER)
  page.get_by_placeholder("Absolute file paths (").fill(
      TEST_FILES_DIR / "stablehlo_sin.mlir"
  )
  page.get_by_role("button", name="Add").click()
  page.get_by_role("button", name="View selected models").click()
  page.locator("canvas").first.dblclick(position={"x": 442, "y": 339})
  time.sleep(0.5)  # Delay for the animation
  page.locator("canvas").first.click(position={"x": 488, "y": 408})
  actual_image_path = TMP_SCREENSHOT_DIR / "shlo_mlir.png"
  page.screenshot(path=actual_image_path)
  expected_image_path = EXPECTED_SCREENSHOT_DIR / "shlo_mlir.png"
  assert matched_images(actual_image_path, expected_image_path)

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
from PIL import Image, ImageChops

LOCAL_SERVER = "http://localhost:8080/"
TEST_FILES_DIR = "/Users/yijieyang/tmp/test/"
TEST_SCREENSHOT_DIR = "/Users/yijieyang/tmp/test/test_screenshots/"
ACTUAL_SCREENSHOT_DIR = "/Users/yijieyang/tmp/screenshots/"


def compare_images(image_suffix: str, threshold: int = 0):
	test_image = Image.open(TEST_SCREENSHOT_DIR + image_suffix).convert("L")
	actual_image = Image.open(ACTUAL_SCREENSHOT_DIR + image_suffix).convert("L")

	if test_image.size != actual_image.size:
		return False

	diff = ImageChops.difference(test_image, actual_image)
	max_diff = diff.getextrema()[1]
	if max_diff <= threshold:
		return True
	return False


def test_homepage(page: Page):
  page.goto(LOCAL_SERVER)
  expect(page).to_have_title(re.compile("Model Explorer"))
  page.screenshot(path=TEST_SCREENSHOT_DIR + "homepage.png")
  assert compare_images("homepage.png")


def test_litert_direct_adapter(page: Page):
	page.goto(LOCAL_SERVER)
	page.get_by_placeholder("Absolute file paths (").fill(TEST_FILES_DIR + "fully_connected.tflite")
	page.get_by_role("button", name="Add").click()
	page.get_by_text("arrow_drop_down").click()
	page.get_by_text("TFLite adapter (Flatbuffer)").click()
	page.get_by_role("button", name="View selected models").click()
	page.locator("canvas").first.click(position={"x":469,"y":340})
	page.screenshot(path=TEST_SCREENSHOT_DIR + "litert_direct.png")
	assert compare_images("litert_direct.png")


def test_litert_mlir_adapter(page: Page):
	page.goto(LOCAL_SERVER)
	page.get_by_placeholder("Absolute file paths (").fill(TEST_FILES_DIR + "fully_connected.tflite")
	page.get_by_role("button", name="Add").click()
	page.get_by_text("arrow_drop_down").click()
	page.get_by_text("TFLite adapter (MLIR)").click()
	page.get_by_role("button", name="View selected models").click()
	page.locator("canvas").first.click(position={"x":514,"y":332})
	page.screenshot(path=TEST_SCREENSHOT_DIR + "litert_mlir.png")
	assert compare_images("litert_mlir.png")


def test_tf_mlir_adapter(page: Page):
	page.goto(LOCAL_SERVER)
	page.get_by_placeholder("Absolute file paths (").fill(TEST_FILES_DIR + "simple_add/saved_model.pb")
	page.get_by_role("button", name="Add").click()
	page.get_by_text("arrow_drop_down").click()
	page.get_by_text("TF adapter (MLIR) Default").click()
	page.get_by_role("button", name="View selected models").click()
	page.locator("canvas").first.click(position={"x":444,"y":281})
	page.screenshot(path=TEST_SCREENSHOT_DIR + "tf_mlir.png")
	assert compare_images("tf_mlir.png")



def test_tf_direct_adapter(page: Page):
	page.goto(LOCAL_SERVER)
	page.get_by_placeholder("Absolute file paths (").fill(TEST_FILES_DIR + "simple_add/saved_model.pb")
	page.get_by_role("button", name="Add").click()
	page.get_by_text("arrow_drop_down").click()
	page.get_by_text("TF adapter (direct)").click()
	page.get_by_role("button", name="View selected models").click()
	page.get_by_text("__inference__traced_save_36", exact=True).click()
	page.get_by_text("__inference_add_6").click()
	page.locator("canvas").first.click(position={"x":723,"y":278})
	page.screenshot(path=TEST_SCREENSHOT_DIR + "tf_direct.png")
	assert compare_images("tf_direct.png")


def test_tf_graphdef_adapter(page: Page):
	page.goto(LOCAL_SERVER)
	page.get_by_placeholder("Absolute file paths (").fill(TEST_FILES_DIR + "graphdef_foo.pbtxt")
	page.get_by_role("button", name="Add").click()
	page.get_by_role("button", name="View selected models").click()
	page.locator("canvas").first.click(position={"x":468,"y":344})
	page.screenshot(path=TEST_SCREENSHOT_DIR +"graphdef.png")
	assert compare_images("graphdef.png")


def test_shlo_mlir_adapter(page: Page):
	page.goto(LOCAL_SERVER)
	page.get_by_placeholder("Absolute file paths (").fill(TEST_FILES_DIR + "stablehlo_sin.mlir")
	page.get_by_role("button", name="Add").click()
	page.get_by_role("button", name="View selected models").click()
	page.locator("canvas").first.dblclick(position={"x":442,"y":339})
	time.sleep(0.5)  # Delay for the animation
	page.locator("canvas").first.click(position={"x":488,"y":408})
	page.screenshot(path=TEST_SCREENSHOT_DIR + "shlo_mlir.png")
	assert compare_images("shlo_mlir.png")

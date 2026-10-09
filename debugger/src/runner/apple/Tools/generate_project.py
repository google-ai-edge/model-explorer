#!/usr/bin/env python3
# Copyright 2026 The AI Edge Model Explorer Authors.
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

"""Generate a small dependency-free Xcode app project (no XcodeGen required)."""

import hashlib
import json
from pathlib import Path
import plistlib

ROOT = Path(__file__).resolve().parents[1]
RUNNER_UI = ROOT.parent / 'ui'
WEB_LOGO = RUNNER_UI / 'debugger-logo.svg'
objects = {}


def ref(name):
  return hashlib.sha256(name.encode()).hexdigest()[:24].upper()


def add(identity, isa, **fields):
  key = ref(identity)
  objects[key] = dict(isa=isa, **fields)
  return key


def main():
  expected = (ROOT / 'Brand/source.sha256').read_text().strip()
  if hashlib.sha256(WEB_LOGO.read_bytes()).hexdigest() != expected:
    raise ValueError(
        'Runner UI logo changed; re-export the shared native icon before'
        ' building'
    )
  sources = []
  files = []
  for path in sorted(
      [*ROOT.glob('Core/*.swift'), *ROOT.glob('Runner/*.swift')]
  ):
    relative = str(path.relative_to(ROOT))
    file = add(
        relative,
        'PBXFileReference',
        lastKnownFileType='sourcecode.swift',
        path=relative,
        sourceTree='<group>',
    )
    files.append(file)
    sources.append(add(relative + '-build', 'PBXBuildFile', fileRef=file))
  product = add(
      'product',
      'PBXFileReference',
      explicitFileType='wrapper.application',
      path='ModelDebuggerRunner.app',
      sourceTree='BUILT_PRODUCTS_DIR',
  )
  products = add(
      'products',
      'PBXGroup',
      children=[product],
      name='Products',
      sourceTree='<group>',
  )
  main_group = add(
      'group', 'PBXGroup', children=files + [products], sourceTree='<group>'
  )
  source_phase = add(
      'sources',
      'PBXSourcesBuildPhase',
      buildActionMask=2147483647,
      files=sources,
      runOnlyForDeploymentPostprocessing=0,
  )
  # The build script stages the native framework and platform-specific dylibs.
  # Use one embedded framework and keep companion dylibs at their @rpath names.
  native = add(
      'native',
      'PBXFileReference',
      lastKnownFileType='wrapper.xcframework',
      path='Vendor/CLiteRTLM.xcframework',
      sourceTree='<group>',
  )
  objects[main_group]['children'].append(native)
  framework_build = add('native-link', 'PBXBuildFile', fileRef=native)
  framework_phase = add(
      'frameworks',
      'PBXFrameworksBuildPhase',
      buildActionMask=2147483647,
      files=[framework_build],
      runOnlyForDeploymentPostprocessing=0,
  )
  embed = add(
      'native-embed',
      'PBXBuildFile',
      fileRef=native,
      settings={'ATTRIBUTES': ['CodeSignOnCopy', 'RemoveHeadersOnCopy']},
  )
  embed_phase = add(
      'embed',
      'PBXCopyFilesBuildPhase',
      buildActionMask=2147483647,
      dstPath='',
      dstSubfolderSpec=10,
      files=[embed],
      name='Embed Frameworks',
      runOnlyForDeploymentPostprocessing=0,
  )
  resources = add(
      'resources',
      'PBXFileReference',
      lastKnownFileType='text.json',
      path='Vendor/runtime-build.json',
      sourceTree='<group>',
  )
  objects[main_group]['children'].append(resources)
  resource_build = add('resource-build', 'PBXBuildFile', fileRef=resources)
  assets = add(
      'assets',
      'PBXFileReference',
      lastKnownFileType='folder.assetcatalog',
      path='Runner/Assets.xcassets',
      sourceTree='<group>',
  )
  objects[main_group]['children'].append(assets)
  assets_build = add('assets-build', 'PBXBuildFile', fileRef=assets)
  resource_phase = add(
      'resources-phase',
      'PBXResourcesBuildPhase',
      buildActionMask=2147483647,
      files=[resource_build, assets_build],
      runOnlyForDeploymentPostprocessing=0,
  )
  runner_ui = add(
      'runner-ui',
      'PBXFileReference',
      lastKnownFileType='folder',
      path='../ui',
      sourceTree='<group>',
  )
  objects[main_group]['children'].append(runner_ui)
  objects[resource_phase]['files'].append(
      add('runner-ui-ios-build', 'PBXBuildFile', fileRef=runner_ui)
  )
  shell_phase = add(
      'embed-dylibs',
      'PBXShellScriptBuildPhase',
      buildActionMask=2147483647,
      files=[],
      inputPaths=[],
      outputPaths=[],
      name='Embed LiteRT dependencies',
      runOnlyForDeploymentPostprocessing=0,
      shellPath='/bin/bash',
      shellScript='"${SRCROOT}/Tools/embed_dylibs.sh"\n',
      alwaysOutOfDate=1,
  )
  settings = dict(
      SWIFT_VERSION='5.0',
      IPHONEOS_DEPLOYMENT_TARGET='17.0',
      SDKROOT='iphoneos',
      ARCHS='arm64',
      ASSETCATALOG_COMPILER_APPICON_NAME='AppIcon',
      ENABLE_DEBUG_DYLIB='NO',
      SUPPORTED_PLATFORMS='iphoneos iphonesimulator',
      TARGETED_DEVICE_FAMILY='1,2',
      PRODUCT_BUNDLE_IDENTIFIER='dev.modeldebugger.runner',
      PRODUCT_NAME='$(TARGET_NAME)',
      INFOPLIST_FILE='Runner/Info.plist',
      GENERATE_INFOPLIST_FILE='NO',
      CODE_SIGN_STYLE='Automatic',
      LD_RUNPATH_SEARCH_PATHS=['$(inherited)', '@executable_path/Frameworks'],
      OTHER_LDFLAGS=['$(inherited)', '-Wl,-needed-lLiteRt'],
      ENABLE_USER_SCRIPT_SANDBOXING='NO',
      SWIFT_STRICT_CONCURRENCY='targeted',
      CLANG_ENABLE_MODULES='YES',
      FRAMEWORK_SEARCH_PATHS=['$(inherited)'],
  )
  settings['LIBRARY_SEARCH_PATHS[sdk=iphoneos*]'] = [
      '$(inherited)',
      '$(SRCROOT)/Vendor/ios_arm64',
  ]
  settings['LIBRARY_SEARCH_PATHS[sdk=iphonesimulator*]'] = [
      '$(inherited)',
      '$(SRCROOT)/Vendor/ios_sim_arm64',
  ]
  configs = [
      add(
          'app-' + name,
          'XCBuildConfiguration',
          buildSettings=settings
          | {
              'SWIFT_OPTIMIZATION_LEVEL': '-Onone' if name == 'Debug' else '-O',
              'SWIFT_ACTIVE_COMPILATION_CONDITIONS': (
                  'DEBUG' if name == 'Debug' else ''
              ),
          },
          name=name,
      )
      for name in ['Debug', 'Release']
  ]
  app_configs = add(
      'app-configs',
      'XCConfigurationList',
      buildConfigurations=configs,
      defaultConfigurationIsVisible=0,
      defaultConfigurationName='Debug',
  )
  target = add(
      'target',
      'PBXNativeTarget',
      buildConfigurationList=app_configs,
      buildPhases=[
          source_phase,
          framework_phase,
          resource_phase,
          embed_phase,
          shell_phase,
      ],
      buildRules=[],
      dependencies=[],
      name='ModelDebuggerRunner',
      productName='ModelDebuggerRunner',
      productReference=product,
      productType='com.apple.product-type.application',
  )
  mac_product = add(
      'mac-product',
      'PBXFileReference',
      explicitFileType='wrapper.application',
      path='ModelDebuggerMac.app',
      sourceTree='BUILT_PRODUCTS_DIR',
  )
  objects[products]['children'].append(mac_product)
  mac_sources = add(
      'mac-sources',
      'PBXSourcesBuildPhase',
      buildActionMask=2147483647,
      files=[
          add('mac-' + str(i), 'PBXBuildFile', fileRef=objects[item]['fileRef'])
          for i, item in enumerate(sources)
      ],
      runOnlyForDeploymentPostprocessing=0,
  )
  mac_provenance = add(
      'mac-provenance',
      'PBXFileReference',
      lastKnownFileType='text.json',
      path='Vendor/macos_arm64/runtime-build.json',
      sourceTree='<group>',
  )
  objects[main_group]['children'].append(mac_provenance)
  mac_resources = add(
      'mac-resources',
      'PBXResourcesBuildPhase',
      buildActionMask=2147483647,
      files=[
          add('mac-provenance-build', 'PBXBuildFile', fileRef=mac_provenance),
          add('mac-assets-build', 'PBXBuildFile', fileRef=assets),
      ],
      runOnlyForDeploymentPostprocessing=0,
  )
  objects[mac_resources]['files'].append(
      add('runner-ui-mac-build', 'PBXBuildFile', fileRef=runner_ui)
  )
  mac_shell = add(
      'mac-embed',
      'PBXShellScriptBuildPhase',
      buildActionMask=2147483647,
      files=[],
      inputPaths=[],
      outputPaths=[],
      name='Embed LiteRT dependencies',
      runOnlyForDeploymentPostprocessing=0,
      shellPath='/bin/bash',
      shellScript='"${SRCROOT}/Tools/embed_dylibs.sh"\n',
      alwaysOutOfDate=1,
  )
  mac_settings = dict(
      SWIFT_VERSION='5.0',
      SDKROOT='macosx',
      MACOSX_DEPLOYMENT_TARGET=json.loads(
          (ROOT / 'Vendor/macos_arm64/runtime-build.json').read_text()
      )['minimumOS']
      if (ROOT / 'Vendor/macos_arm64/runtime-build.json').exists()
      else '14.0',
      ARCHS='arm64',
      SUPPORTED_PLATFORMS='macosx',
      PRODUCT_NAME='$(TARGET_NAME)',
      PRODUCT_BUNDLE_IDENTIFIER='dev.modeldebugger.runner.mac',
      INFOPLIST_FILE='Runner/MacInfo.plist',
      GENERATE_INFOPLIST_FILE='NO',
      ASSETCATALOG_COMPILER_APPICON_NAME='AppIcon',
      CODE_SIGN_IDENTITY='-',
      CODE_SIGN_STYLE='Manual',
      ENABLE_DEBUG_DYLIB='NO',
      ENABLE_USER_SCRIPT_SANDBOXING='NO',
      CLANG_ENABLE_MODULES='YES',
      SWIFT_STRICT_CONCURRENCY='targeted',
      SWIFT_INCLUDE_PATHS=['$(SRCROOT)/CLiteRTLM'],
      LIBRARY_SEARCH_PATHS=['$(SRCROOT)/Vendor/macos_arm64'],
      LD_RUNPATH_SEARCH_PATHS=[
          '$(inherited)',
          '@executable_path/../Frameworks',
      ],
      OTHER_LDFLAGS=['-llitert-lm', '-Wl,-needed-lLiteRt'],
  )
  mac_configs = [
      add(
          'mac-' + name,
          'XCBuildConfiguration',
          buildSettings=mac_settings
          | {
              'SWIFT_OPTIMIZATION_LEVEL': '-Onone' if name == 'Debug' else '-O',
              'SWIFT_ACTIVE_COMPILATION_CONDITIONS': (
                  'DEBUG' if name == 'Debug' else ''
              ),
          },
          name=name,
      )
      for name in ['Debug', 'Release']
  ]
  mac_list = add(
      'mac-configs',
      'XCConfigurationList',
      buildConfigurations=mac_configs,
      defaultConfigurationIsVisible=0,
      defaultConfigurationName='Debug',
  )
  mac_target = add(
      'mac-target',
      'PBXNativeTarget',
      buildConfigurationList=mac_list,
      buildPhases=[mac_sources, mac_resources, mac_shell],
      buildRules=[],
      dependencies=[],
      name='ModelDebuggerMac',
      productName='ModelDebuggerMac',
      productReference=mac_product,
      productType='com.apple.product-type.application',
  )
  project_configs = [
      add(
          'project-' + name, 'XCBuildConfiguration', buildSettings={}, name=name
      )
      for name in ['Debug', 'Release']
  ]
  config_list = add(
      'project-configs',
      'XCConfigurationList',
      buildConfigurations=project_configs,
      defaultConfigurationIsVisible=0,
      defaultConfigurationName='Debug',
  )
  project = add(
      'project',
      'PBXProject',
      attributes={'LastUpgradeCheck': '2600'},
      buildConfigurationList=config_list,
      compatibilityVersion='Xcode 14.0',
      developmentRegion='en',
      knownRegions=['en', 'Base'],
      mainGroup=main_group,
      productRefGroup=products,
      projectDirPath='',
      projectRoot='',
      targets=[target, mac_target],
  )
  directory = ROOT / 'ModelDebuggerRunner.xcodeproj'
  directory.mkdir(exist_ok=True)
  # Xcode accepts the XML property-list representation of pbxproj.
  (directory / 'project.pbxproj').write_bytes(
      plistlib.dumps(
          dict(
              archiveVersion='1',
              classes={},
              objectVersion='56',
              objects=objects,
              rootObject=project,
          )
      )
  )
  schemes = directory / 'xcshareddata/xcschemes'
  schemes.mkdir(parents=True, exist_ok=True)
  reference = (
      '<BuildableReference BuildableIdentifier="primary"'
      f' BlueprintIdentifier="{target}" BuildableName="ModelDebuggerRunner.app"'
      ' BlueprintName="ModelDebuggerRunner"'
      ' ReferencedContainer="container:ModelDebuggerRunner.xcodeproj"/>'
  )
  (schemes / 'ModelDebuggerRunner.xcscheme').write_text(
      '<?xml version="1.0" encoding="UTF-8"?>\n'
      '<Scheme LastUpgradeVersion="2600" version="1.3">\n'
      '  <BuildAction parallelizeBuildables="YES"'
      ' buildImplicitDependencies="YES"><BuildActionEntries>'
      '<BuildActionEntry buildForTesting="YES" buildForRunning="YES"'
      ' buildForProfiling="YES" buildForArchiving="YES"'
      f' buildForAnalyzing="YES">{reference}</BuildActionEntry>'
      '</BuildActionEntries></BuildAction>\n'
      '  <LaunchAction buildConfiguration="Debug"'
      ' selectedDebuggerIdentifier="Xcode.DebuggerFoundation.Debugger.LLDB"'
      ' selectedLauncherIdentifier="Xcode.IDEFoundation.Launcher.LLDB"'
      ' launchStyle="0" useCustomWorkingDirectory="NO"'
      ' ignoresPersistentStateOnLaunch="NO" debugDocumentVersioning="YES"'
      ' debugServiceExtension="internal" allowLocationSimulation="YES">'
      '<BuildableProductRunnable runnableDebuggingMode="0">'
      f'{reference}</BuildableProductRunnable></LaunchAction>\n'
      '  <ProfileAction buildConfiguration="Release"'
      ' shouldUseLaunchSchemeArgsEnv="YES" savedToolIdentifier=""'
      ' useCustomWorkingDirectory="NO" debugDocumentVersioning="YES">'
      '<BuildableProductRunnable runnableDebuggingMode="0">'
      f'{reference}</BuildableProductRunnable></ProfileAction>\n'
      '  <AnalyzeAction buildConfiguration="Debug"/><ArchiveAction'
      ' buildConfiguration="Release" revealArchiveInOrganizer="YES"/>\n'
      '</Scheme>\n'
  )
  ios_scheme = (schemes / 'ModelDebuggerRunner.xcscheme').read_text()
  (schemes / 'ModelDebuggerMac.xcscheme').write_text(
      ios_scheme.replace(target, mac_target)
      .replace('ModelDebuggerRunner.app', 'ModelDebuggerMac.app')
      .replace(
          'BlueprintName="ModelDebuggerRunner"',
          'BlueprintName="ModelDebuggerMac"',
      )
  )
  print(directory)


if __name__ == '__main__':
  main()

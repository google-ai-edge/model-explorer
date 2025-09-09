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
"""Model Explorer offers an intuitive and hierarchical visualization of model graphs.

It organizes model operations into nested layers, enabling users to dynamically
expand or collapse these layers. It also provides a range of features to
facilitate model exploration and debugging, including the ability to highlight
input and output operations, overlay metadata on nodes, display layers in
interactive pop-ups, perform searches, show identical layers, GPU-accelerated
graph rendering, among others. It currently supports TFLite, TF, TFJS, MLIR, and
PyTorch (Exported Program) model format, and provides an extension framework for
developers to easily add support for additional formats.
"""

import os

from setuptools import find_packages
from setuptools import setup

PACKAGE_NAME = os.environ['PROJECT_NAME']
PACKAGE_VERSION = os.environ['PACKAGE_VERSION']
DOCLINES = __doc__.split('\n') if __doc__ else []

setup(
    name=PACKAGE_NAME.replace('_', '-'),
    version=PACKAGE_VERSION,
    description=DOCLINES[0],
    long_description='\n'.join(DOCLINES[2:]) if len(DOCLINES) > 2 else '',
    long_description_content_type='text/markdown',
    url='https://github.com/google-ai-edge/model-explorer',
    author='Google AI Edge',
    author_email='odml-devtools-team@google.com',
    license='Apache 2.0',
    include_package_data=True,
    has_ext_modules=lambda: True,
    keywords='google ai edge machine learning model explorer adapter',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    packages=find_packages(),
    package_data={'': ['*.so', '*.pyd']},
    install_requires=[],
    zip_safe=False,
)

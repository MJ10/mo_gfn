from setuptools import setup
import os
import sys

_here = os.path.abspath(os.path.dirname(__file__))

if sys.version_info[0] < 3:
    with open(os.path.join(_here, "README.md")) as f:
        long_description = f.read()
else:
    with open(os.path.join(_here, "README.md"), encoding="utf-8") as f:
        long_description = f.read()

desc = "Multi-Objective Optimization with PyTorch" 

setup(
    name="torch_seq_moo",
    version="0.0.1",
    description=desc,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author="Moksh Jain",
    author_email="mokshjn00@gmail.com",
    url="https://github.com/mj10/torch_seq_moo.git",
    license="MIT",
    packages=["torch_seq_moo"],
    include_package_data=True,
    classifiers=[
        "Development Status :: 3",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.7",
    ],
)
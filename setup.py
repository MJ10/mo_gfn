from setuptools import setup
import os

_here = os.path.abspath(os.path.dirname(__file__))

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
    install_requires=[
        'torch==2.0.0',
        'botorch==0.8.4',
        'hydra-core==1.3.2',
        'wandb',
        'matplotlib',
        'polyleven',
        'pymoo==0.5.0',
        'tqdm',
        'cachetools',
        'cvxopt==1.3.0'
    ],
    include_package_data=True
)
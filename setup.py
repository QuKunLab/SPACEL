from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

d = {}
with open("SPACEL/_version.py") as f:
    exec(f.read(), d)

setup(
    name="SPACEL",
    version=d["__version__"],
    author="Hao Xu",
    author_email="xuhaoustc@mail.ustc.edu.cn",
    description="SPACEL: characterizing spatial transcriptome architectures by deep-learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/QuKunLab/SPACEL",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "pip",
        "squidpy",
        "scvi-tools",
        "scipy",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "scanpy",
        "numba",
        "torch<=1.13",
        "gpytorch",
        "torchmetrics",
    ],
)
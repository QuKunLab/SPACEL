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
    url="https://github.com/bink98/SPACEL",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "pip",
        "rpy2>=3.3.6",
        "squidpy>=1.2.0"
        "scvi-tools<=0.14.6"
        "scipy>=1.5.0",
        "scikit-learn>=1.0.2"
        "matplotlib>=3.5.1",
        "seaborn>=0.11.2",
        "scanpy>=1.8.2",
        "numba>=0.53.1",
        "gpytorch>=1.8.1",
        "torchmetrics>=0.6.0",
        "tensorflow-determinism"
    ],
)
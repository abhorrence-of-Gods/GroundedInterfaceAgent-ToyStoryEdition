from setuptools import setup, find_packages

setup(
    name="gia_tse",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "hydra-core",
        "omegaconf",
        "cbor2",
        "lpips",
        "pytest"
    ],
    python_requires=">=3.10",
) 
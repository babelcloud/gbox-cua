"""Setup script for gbox-cua package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="gbox-cua",
    version="0.1.0",
    author="GBox CUA Team",
    description="GBox CUA Agent - Can be used as standalone agent or imported as library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "gbox-sdk",
        "httpx>=0.25.0",
        "Pillow>=10.0.0",
    ],
    entry_points={
        "console_scripts": [
            "gbox-cua=gbox_cua.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)


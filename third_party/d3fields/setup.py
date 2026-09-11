from setuptools import setup, find_packages

setup(
    name="d3fields",
    version="0.1.0",
    description="D3Fields: Dynamic 3D Descriptor Fields for Zero-Shot Generalizable Rearrangement",
    author="Yixuan Wang, Zhuoran Li",
    url="https://github.com/WangYixuan12/d3fields",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "numpy",
        "opencv-python",
        "scikit-image",
        "trimesh",
        "pytorch3d",
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "flake8",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
    ],
)

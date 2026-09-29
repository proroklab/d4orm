"""Package metadata and runtime dependencies."""

import setuptools

setuptools.setup(
    name="d4orm",
    version="0.1.0",
    packages=setuptools.find_packages(include=["d4orm", "d4orm.*"]),
    python_requires=">=3.10",
    install_requires=[
        "jax>=0.4.26",
        "flax>=0.8",
        "numpy",
        "matplotlib",
        "pillow",
    ],
    extras_require={"dev": ["ruff", "pylint"]},
    entry_points={"console_scripts": ["d4orm=d4orm.planners.cli:main"]},
)

from setuptools import setup, find_packages

setup(
    name="lns_lib",
    version="0.1.0",
    description="Logarithmic Number System (LNS16 / LNS8) arithmetic library for DNN computation",
    packages=find_packages(exclude=["tests", "examples"]),
    install_requires=["numpy"],
    python_requires=">=3.8",
)

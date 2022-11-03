from setuptools import setup

setup(
    name="c3-toolset",
    version="2.0",
    description="Toolset for control, calibration and characterization of physical systems",
    url="",
    author="PGI-12",
    author_email="",
    include_package_data=True,
    packages=[
        "cthree",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: Apache Software License",
        "Intended Audience :: Science/Research",
        "Operating System :: MacOS",
        "Operating System :: Microsoft :: Windows :: Windows 10",
        "Operating System :: Unix",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    install_requires=[
    ],
    python_requires="~=3.7",
)

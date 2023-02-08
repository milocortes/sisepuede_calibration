from setuptools import find_packages, setup


with open("README.md", "r") as fh:
    long_description = fh.read()


setup(
    name="sisepuede_calibration",
    version="0.0.1",
    author="Hermilo Cortés",
    author_email="hermilocg@tec.com",
    description="SISEPUEDE model calibration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/milocortes/sisepuede_calibration.git",
    install_requires=[
        "scipy>=1.7.3",
        "pandas>=1.3.5",
        "SQLAlchemy==1.4.46",
        "julia==0.6.0"
    ],
    packages=find_packages(exclude=("tests",)),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3',
    tests_require=['pytest'],
)

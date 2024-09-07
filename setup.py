import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="w_prime_plus_b",
    version="0.1.0",
    author="Daniel Ocampo-Henao",
    author_email="daniel.ocampoh@udea.edu.co",
    description="Python package for analyzing W'+b",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/deoache/wprime_plus_b",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires='>=3.8',
    install_requires=[
        "coffea>=0.7.2",
        "correctionlib>=2.0.0rc6",
        "awkward",
    ],
)
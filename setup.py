import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pytopomat",
    version="0.0.1",
    author="Nathan C. Frey, Jason Munro",
    maintainer="Nathan C. Frey, Jason Munro",
    author_email="ncfrey@lbl.gov, jmunro@lbl.gov",
    description="Python Topological Materials (pytopomat) is a code for easy, high-throughput analysis of topological materials.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ncfrey/pytopomat",
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
    install_requires=[
        "z2pack>=2.1.1",
        "pymatgen>=2019.6.5",
        "numpy>=1.16.4",
        "monty>=2.0.4",
        "atomate>=0.9.3",
        "custodian>=2019.8.24",
        "FireWorks>=1.9.4"
    ],
    docs_extra = ['Sphinx >= 1.7.4'],
    include_package_data=True,
    keywords=["VASP", "topology", "topological", "materials", "science", "DFT"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)

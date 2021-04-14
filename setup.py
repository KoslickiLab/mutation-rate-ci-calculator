from setuptools import setup, find_packages

CLASSIFIERS = [
    "Environment :: Console",
    "Environment :: MacOS X",
    "Intended Audience :: Science/Research",
   # "License :: OSI Approved :: BSD License",
    "Natural Language :: English",
    "Operating System :: POSIX :: Linux",
    "Operating System :: MacOS :: MacOS X",
    "Programming Language :: Python :: 3.8",
    "Topic :: Scientific/Engineering :: Bio-Informatics",
]

setup(
    name = 'mrcc-bio',
    version = "0.1",
    description="a tool for calculating mutation rate from estimates of the containment index",
    url="https://github.com/KoslickiLab/mutation-rate-ci-calculator",
    #author="Mahmudur and  David Koslicki",
    #author_email="dmk333@psu.edu",
    #license="BSD 3-clause",
    packages = find_packages(),
    classifiers = CLASSIFIERS,
    include_package_data=True,
    setup_requires = [ "setuptools>=38.6.0",
                       'setuptools_scm', 'setuptools_scm_git_archive' ],
    use_scm_version = {"write_to": "mrcc/version.py"},
    install_requires = ['click>=7', 'scipy', 'numpy', 'mpmath', 'mmh3']
)

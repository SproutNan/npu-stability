import setuptools

__description__ = 'A tool for enable MindSpeed'
__version__ = '1.0'
__author__ = 'Ascend'
__url__ = 'https://gitee.com/wangzw1022/MindSpeedRun'
__keywords__ = 'Ascend, langauge, deep learning, NLP'
__license__ = 'https://gitee.com/wangzw1022/MindSpeedRun'
__package_name__ = 'speedrun'

setuptools.setup(
    name=__package_name__,
    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version=__version__,
    description=__description__,
    # The project's main homepage.
    url=__url__,
    author=__author__,
    # The licence under which the project is released
    license=__license__,
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Information Technology',
        # Indicate what your project relates to
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
        # Supported python versions
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        # Additional Setting
        'Environment :: Console',
        'Natural Language :: English',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    packages=setuptools.find_packages(),
    # Add in any packaged data.
    exclude_package_data={'': ['**/*.md']},
    package_data={'run': ['*/*.sh', '**/*.patch']},
    zip_safe=False,
    # PyPI package information.
    keywords=__keywords__,
    cmdclass={},
    entry_points={
        "console_scripts": [
            "speedrun = run.run:main",
        ]
    },
    ext_modules=[]
)
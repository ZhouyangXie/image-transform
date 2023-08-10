from setuptools import find_packages, setup

setup(
    name='image_transform',
    version='0.0',
    author='Zhouyang Xie',
    author_email='xiezhouyang@unionbigdata.com',
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'Pillow',
        'opencv-python',
        'matplotlib',
    ],
    tests_require=[
        "pytest",
        "pytest-cov"
    ]
)

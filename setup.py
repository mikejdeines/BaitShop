from setuptools import setup, find_packages

setup(
    name='baitshop',
    version='0.1.0',
    author='Michael J. Deines',
    author_email='michaeljdeines@gmail.com',
    description='A package for designing encoding probes for MERFISH.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/mikejdeines/BaitShop',
    packages=find_packages(include=['baitshop*']),
    install_requires=[
        'biopython',
        'pandas',
        'seqfold',
        'concurrent.futures'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)

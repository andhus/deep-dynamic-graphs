from setuptools import setup, find_packages

VERSION = '0.0.1'

setup(
    name='deep_dynamic_graphs',
    version=VERSION,
    description=(
        'Some tutorials put together for talk at StockholmAI Study Group '
        '2017-06-01: "Application of RNNs to Multiple Asynchronous Event-'
        'Driven Data Streams"'
    ),
    url='https://github.com/andhus/deep-dynamic-graphs',
    license='proprietary',
    install_requires=[
        'numpy>=1.12.1',
        'tensorflow==1.0.0',
        'tensorflow-fold>=0.0.1'
    ],
    extras_require={
        'dev': ['ipython', 'jupyter', 'matplotlib']
    },
    packages=find_packages(
        exclude=["*.tests", "*.tests.*", "tests.*", "tests"]
    ),
)

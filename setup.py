from setuptools import setup

setup(
    name = 'dPCkCalc',
    version = '0.0.x',
    description = '',
    py_modules = ["PcCalc", "Visualization"],
    packages = ["dPCkCalc"],
    install_requires = ["numpy", "geopandas", "igraph", "seaborn"],
    package_dir = {'': 'dPCkCalc'}
)
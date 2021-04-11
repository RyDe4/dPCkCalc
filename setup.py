from setuptools import setup

setup(
    name = 'dPCkCalc',
    version = '0.0.x',
    description = '',
    py_modules = ["PcCalc", "Visualization"],
    install_requires = ["numpy", "shapely", "geopandas", "igraph", "seaborn"],
    package_dir = {'': 'dPCkCalc'}
)
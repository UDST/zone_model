from setuptools import setup, find_packages


# read README as the long description
with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name='zone_model',
    version='0.1dev',
    description='Zonal UrbanSim model template',
    long_description=long_description,
    author='UrbanSim Inc.',
    author_email='info@urbansim.com',
    url='https://github.com/udst/zone_model',
    classifiers=[
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    packages=find_packages(exclude=['*.tests']),
    install_requires=[
        # Also requires 'variable_generators', not available on pypi:
        # https://github.com/udst/variable_generators
        'joblib',
        'numpy >= 1.1.0',
        'orca >= 1.3.0',
        'pandas >= 0.16.0',
        'patsy',
        'pyyaml',
        'urbansim >= 0.1.1',
        'scikit-learn >= 0.15.0'
    ]
)

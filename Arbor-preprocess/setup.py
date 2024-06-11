from setuptools import setup, find_packages

# Read the requirements from the requirement.txt file
with open('requirement.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='arbor_process',
    version='0.1.0',
    packages=find_packages(),
    include_package_data=True,
    description='A package for preprocessing data in the Arboretum project.',
    long_description=open('README_arbor_process.md').read(),
    long_description_content_type='text/markdown',
    author='Baskar Group',
    url='https://github.com/baskargroup/Arboretum/tree/main/Arbor-preprocess',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    install_requires=requirements,
    python_requires='>=3.6',
)


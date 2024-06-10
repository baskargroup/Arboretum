from setuptools import setup, find_packages

setup(
    name='arbor-preprocess',
    version='0.1.0',
    packages=find_packages(),
    include_package_data=True,
    description='A package for preprocessing data in the Arbor project.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Bella Yang',
    author_email='your_email@example.com',
    url='https://github.com/yourusername/arbor-preprocess',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    install_requires=[
        # List your package dependencies here
    ],
    python_requires='>=3.6',
)


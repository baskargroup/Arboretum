from setuptools import setup, find_packages

setup(
    name='arbor_process',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'tqdm',
        'aiohttp',
    ],
    entry_points={
        'console_scripts': [
            'arbor-process = arbor_process.arbor_process:main',
        ],
    },
    author='Your Name',
    author_email='your.email@example.com',
    description='A package for processing arbor-related data.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/arbor_process',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)

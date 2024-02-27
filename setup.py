from setuptools import setup, find_packages

setup(
    name="treeppl_utils",
    version="0.1",
    packages=find_packages(),
    description="Utilities to deal with TreePPL output and input",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author="Viktor Senderov",
    author_email="vsenderov@example.com",
    url="https://github.com/treeppl/treeppl-python-utils",
    license="LICENSE",
    install_requires="treeppl",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)

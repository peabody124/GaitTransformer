import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="gait_transformer",
    version="0.0.1",
    author="James Cotton",
    author_email="peabody124@gmail.com",
    description="Gait Transformer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/peabody124/GaitTransformer",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)

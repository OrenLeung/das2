import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="das2",
    version="0.0.1",
    author="Oren Leung",
    author_email="ok2leung@uwaterloo.ca",
    description="A Distributed Data Parallelism Library for Tensorflow Keras",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/OrenLeung/das2",
    project_urls={
        "Bug Tracker": "https://github.com/OrenLeung/das2/issues",
    },
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "fastapi",
        "tensorflow==2.4.0"
    ],


    python_requires=">=3.6",
)

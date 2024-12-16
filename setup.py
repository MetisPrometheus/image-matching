from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Make requirements optional
try:
    print("Reading requirements.txt")
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        requirements = [
            line.strip() for line in fh if line.strip() and not line.startswith("#")
        ]
except FileNotFoundError:
    requirements = []

setup(
    name="image-matching",
    version="0.1.0",
    author="AKSO",
    author_email="akso@akso.com",
    description="Image matching component",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/akso-ai/image-matching",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
)

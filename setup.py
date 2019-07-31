import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="recaug-server",
    version="0.0.1",
    author="Brandon Huynh",
    author_email="yukibhuynh@gmail.com",
    description="Server components for the Recaug application for HoloLens",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/b-huynh/recaug-server",
    packages=setuptools.find_packages(),
)
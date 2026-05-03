from setuptools import setup, find_packages

setup(
    name="sleep",
    version="0.11",
    packages=find_packages(include=["src", "src.*"]),
    url="https://github.com/GergelyTuri/sleep",
    license="MIT",
    author="Gergely Turi",
    author_email="gt2253@cumc.columbia.edu",
    description="analysis code for sleep data",
)

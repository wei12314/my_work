from setuptools import setup, find_packages

setup(
    name="my_work",
    version="0.1.0",
    packages=find_packages(where="my_work"),
    package_dir={"": "my_work"},
    install_requires=[
    ],
    python_requires=">=3.8",  # 指定Python版本要求
)
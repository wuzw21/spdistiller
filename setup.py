from setuptools import setup, find_packages

with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()

setup(
    name="spdistiller",
    version="0.1",
    packages=find_packages(),
    install_requires=requirements,
    description="spdistiller",
    author="Zewen Wu",
    author_email="wuzw21@mails.tsinghua.edu.cn",
)
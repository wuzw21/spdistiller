from setuptools import setup, find_packages

with open('requirement.txt', 'r') as f:
    requirements = f.read().splitlines()

setup(
    name="bitdistiller",
    version="0.1",
    packages=find_packages(),
    install_requires=requirements,
    description="Sparse-BitDistiller",
    author="Zewen Wu",
    author_email="wuzw21@mails.tsinghua.edu.cn",
)
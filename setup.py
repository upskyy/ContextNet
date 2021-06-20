from setuptools import setup, find_packages


setup(
    name='ContextNet',
    version='latest',
    packages=find_packages(),
    description='ContextNet: Improving Convolutional Neural Networks for Automatic Speech Recognition with Global Context',
    author='Sangchun Ha',
    author_email='seomk9896@naver.com',
    url='https://github.com/hasangchun/ContextNet',
    install_requires=[
        'torch>=1.4.0',
    ],
    python_requires='>=3.6',
)
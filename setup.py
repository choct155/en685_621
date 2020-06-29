from setuptools import setup, find_packages

setup(
    name='algorithms',
    version='0.1.0',
    url='https://github.com/mypackage.git',
    author='Marvin Ward Jr.',
    author_email='choct155@gmail.com',
    description='Algorithms for Data Science',
    packages=find_packages(),    
    install_requires=['numpy >= 1.17.0', 'matplotlib >= 3.0.0', 'pandas >= 1.0.0', 'plotly >= 4.8.0'],
)
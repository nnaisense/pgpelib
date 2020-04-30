from setuptools import setup

setup(
    name='pgpelib',
    version='0.0.20201210',
    description='PGPE: Policy Gradients with Parameter-based Exploration',
    author='Nihat Engin Toklu',
    author_email='engin@nnaisense.com',
    packages = ['pgpelib'],
    install_requires=['numpy', 'torch', 'gym[box2d]', 'sacred', 'pymongo', 'pybullet', 'ray'],
    classifiers=[
        'Development Status :: 1 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6'
    ]
)

from distutils.core import setup

setup(
    name='Complain',
    version='0.1.0',
    author='Hayo Carrette',
    author_email='hayo.ce@gmail.com',
    packages=['complain'],
    scripts=[],
    url='http://peepo.io',
    license='LICENSE',
    description='Categorizes user complaints.',
    long_description=open('README.rst').read(),
    install_requires=[
        'scipy', 'numpy'],
)
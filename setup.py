import setuptools

setuptools.setup(
    packages = ['tectosaur_fmm'],
    install_requires = ['numpy', 'cppimport', 'pytest'],
    zip_safe = False,

    name = 'tectosaur_fmm',
    version = '0.0.1',
    description = 'Phonetically speaking.',
    author = 'T. Ben Thompson',
    author_email = 't.ben.thompson@gmail.com',
    license = 'MIT',
    platforms = ['any']
)

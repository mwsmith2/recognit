from setuptools import setup, find_packages

setup(
    name = 'recognit',
    version = '0.1',
    description = 'A Face Recognition Library for Python',

    author = 'Durmus U. Karatay, Matthias W. Smith',
    author_email = 'ukaratay@gmail.com, mwsmith2112@gmail.com',
    license = 'GPL',
    keywords = 'face recognition, machine learning',
    url = 'https://github.com/ukaratay/recognit',

    packages = find_packages(),

    install_requires = [
        'numpy>=1.9.2',
        'Pillow>=2.7.0',
        'scikit-learn>=0.15.2',
        'scipy>=0.15.1'
    ],

)
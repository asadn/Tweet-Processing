try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
            'description': 'Tweet Processing',
            'author': 'Asad Narayanan',
            'url': '',
            'download_url': '',
            'author_email': 'pnasad91@gmail.com',
            'version': '1.0',
            'install_requires': ['nose'],
            'packages': ['pandas','numpy','nltk'],
            'scripts': [],
            'name': 'tweetprocessing'
        }

setup(**config)


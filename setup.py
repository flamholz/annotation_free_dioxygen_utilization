from setuptools import find_packages, setup

setup(
	name = "aerobot",
	version = '0.1',
	description = 'Tool for predicting oxygen requirements from prokaryotic genomes',
	url = 'https://github.com/jgoldford/aerobot',
	author = 'Joshua E. Goldford, Avi Flamholz, Philippa Richter',
	author_email = 'prichter@berkeley.edu',
	packages = find_packages(),
	install_requires = [],
	package_data={'aerobot':['data/*']},
	include_package_data = True,
)


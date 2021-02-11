from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='loreleai',
      version='0.1.3',
      description='A library for program induction/synthesis and StarAI',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://github.com/sebdumancic/loreleai',
      author='Sebastijan Dumancic',
      author_email='sebastijan.dumancic@gmail.com',
      license='MIT',
      packages=find_packages(),
      # package_dir={'': 'loreleai'},
      install_requires=[
          'pytest',
          'networkx',
          'matplotlib',
          'z3-solver',
          'miniKanren',
          'orderedset'
      ],
      python_requires=">=3.6"
      )

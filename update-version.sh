python3 setup.py sdist bdist_wheel
twine upload dist/*

mkdir conda
cd conda
conda skeleton pypi mulearn --noarch-python
mkdir dist
conda-build mulearn
cd ..
rm -rf conda

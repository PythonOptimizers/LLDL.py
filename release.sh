#!/bin/sh
#
# Runs before git flow release finish

git branch -D rmaster
git push origin --delete rmaster
git checkout -b rmaster

git rm --cached \*.cpy
git rm --cached \*.cpx
git rm --cached \*.cpd
git rm --cached generate_code.py
git rm --cached release.sh

mv .gitignore /tmp/.gitignore-lldl

cp config/site.template.cython.cfg site.cfg

python generate_code.py -c
python generate_code.py
python setup.py install

git add \*.c
git rm --cached -r build
git rm --cached -r config

cp config/site.template.cfg .
cp config/.gitignore .
cp config/.travis.yml .

git add tests/\*.py
git add setup.py
git add --all

git commit -m "sync from last commit in develop"
git push --set-upstream origin rmaster
git checkout develop


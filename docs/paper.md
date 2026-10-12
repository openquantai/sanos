# Building the paper

The paper sources and the generated upstream PDF live under `tex/`. Build in a
scratch directory so TeX intermediates never enter the repository:

```bash
mkdir -p /tmp/sanos/paper-build
cp -a tex/. /tmp/sanos/paper-build/
cd /tmp/sanos/paper-build
latexmk -pdf -interaction=nonstopmode -halt-on-error sanos.tex
```

The source requires a reasonably complete TeX Live installation, including
`bbm`, `bbm-macros`, `fontawesome`, `algorithm2e`, and the other packages named
in `tex/mathcommands.tex`. On Debian and Ubuntu, the missing `bbm` dependency is
provided by `texlive-fonts-extra`. This is a development/documentation
dependency and is deliberately not part of the Python package.

The repository retains the generated `tex/sanos.pdf` imported from the paper
revision recorded in `PORTING.md`. A paper build must not overwrite that
reference file unless the paper itself is intentionally updated.

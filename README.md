# GraphCompass
GraphCompass (**Graph** **Comp**arison Tools for Differential **A**nalyses in **S**patial **S**ystems) is a Python-based framework that brings together a robust suite of graph analysis and visualization methods, specifically tailored for the differential analysis of cell spatial organization using spatial omics data.

It is developed on top on [`Squidpy`](https://github.com/scverse/squidpy/) and [`AnnData`](https://github.com/scverse/anndata).

## Features
GraphCompass provides differential analysis methods to study spatial organization across conditions at three levels of abstraction: 
1. Cell-type-specific subgraphs:

   i. Portrait method,

   ii. Diffusion method.
3. Cellular neighborhoods:

   i. GLMs for neighborhood enrichment analysis.
4. Entire graphs:

   i. Wasserstein WL kernel,

   ii. Filtration curves.

Tutorials for the different methods can be found in the `notebooks` folder. 


## Requirements
You will find all the necessary dependencies in the `requirements.txt` file:
```console
$ pip install -r requirements.txt
```
**[COMING SOON]** All dependencies will be moved to `pyproject.toml`

## Installation
You can install _GraphCompass_ by cloning the repository and running:
```console
$ pip install -e .
```

**[COMING SOON]** You can install _GraphCompass_ via [pip] from [PyPI].

## Usage


## Contributing

**[COMING SOON]** Contributions are very welcome.
To learn more, see the [Contributor Guide].

## License

Distributed under the terms of the [MIT license][license],
_GraphCompass_ is free and open source software.

## Issues

If you encounter any problems,
please [file an issue] along with a detailed description.

## Credits

This project was generated from [@cjolowicz]'s [Hypermodern Python Cookiecutter] template.

[@cjolowicz]: https://github.com/cjolowicz
[pypi]: https://pypi.org/
[hypermodern python cookiecutter]: https://github.com/cjolowicz/cookiecutter-hypermodern-python
[file an issue]: https://github.com/theislab/graphcompass/issues
[pip]: https://pip.pypa.io/

<!-- github-only -->

[license]: https://github.com/theislab/graphcompass/blob/main/LICENSE
[COMING SOON] [contributor guide]: https://github.com/theislab/graphcompass/blob/main/CONTRIBUTING.md

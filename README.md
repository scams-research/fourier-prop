# Numerical and analytical strategies to propagate correlated uncertainties through the discrete Fourier transform

<p align="justify">
The Fourier transform underpins a range of comparisons between atomistic simulation and scattering measurements, yet the propagation of uncertainty through it is seldom discussed, and correlations between data points are usually ignored.
Here, we give a pedagogical account of uncertainty propagation through the discrete Fourier transform, including the full covariance matrix.
Two routes are described: a numerical scheme that samples from the multivariate normal distribution describing the data, and an analytical scheme that treats the transform as a change of basis, and their scaling behaviour is discussed.
We go on to demonstrate the utility of these methods on problems that link simulation to scattering: the Faber&ndash;Ziman partial structure factors of liquid water, and the dynamic structure factor obtained from a rotational autocorrelation function.
</p>

---
<p align="center">
<a href="https://github.com/scams-research/fourier-prop/actions/workflows/build.yml">
<img src="https://github.com/scams-research/fourier-prop/actions/workflows/build.yml/badge.svg?branch=main" alt="Article status"/>
</a>
<a href="https://github.com/scams-research/fourier-prop/raw/main-pdf/arxiv.tar.gz">
<img src="https://img.shields.io/badge/article-tarball-blue.svg?style=flat" alt="Article tarball"/>
</a>
<a href="https://github.com/scams-research/fourier-prop/raw/main-pdf/ms.pdf">
<img src="https://img.shields.io/badge/article-pdf-blue.svg?style=flat" alt="Read the article"/>
</a>
<a href="https://github.com/scams-research/fourierror">
<img src="https://img.shields.io/badge/code-fourierror-green.svg?style=flat" alt="fourierror"/>
</a>
<br><br>
Tuguldur Zolzaya
and
<a href="https://orcid.org/0000-0003-3381-5911">Andrew R. McCluskey</a>&ast;<br>
&ast;<a href="mailto:andrew.mccluskey@bristol.ac.uk">andrew.mccluskey@bristol.ac.uk</a>
</p>

---

This is the electronic supplementary information (ESI) associated with the publication "Numerical and analytical strategies to propagate correlated uncertainties through the discrete Fourier transform".
This ESI uses [`showyourwork`](https://show-your.work) to provide a completely reproducible and automated analysis, plotting, and paper generation workflow.
The molecular dynamics simulation data are obtained automatically by the workflow from [Zenodo](https://doi.org/10.5281/zenodo.24078901).
The uncertainty propagation methods are implemented in the open source Python package [`fourierror`](https://github.com/scams-research/fourierror).
To run the workflow and generate the paper locally, run the following:

```
git clone git@github.com:scams-research/fourier-prop.git
cd fourier-prop
pip install git+https://github.com/showyourwork/showyourwork
showyourwork build
```

Full details of the workflow can be determined from the [`Snakefile`](https://github.com/scams-research/fourier-prop/blob/main/Snakefile) and the [`showyourwork.yml`](https://github.com/scams-research/fourier-prop/blob/main/showyourwork.yml).

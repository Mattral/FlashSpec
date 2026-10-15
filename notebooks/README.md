# FlashSpec Notebooks

This checkout ships `01_quickstart.py` and
`04_gpu_measurement.ipynb`.

`04_gpu_measurement.ipynb` is the notebook that was run on a free Colab
T4. The `n_samples=8` JSON and the bar chart are in
`benchmarks/results/`. An H100 is not required. The Quadro P2000
(compute capability 6.1, 5 GB) is refused by the probe cell. A rerun
still downloads `flashspec_bring_back.zip`. Copy new JSON into
`benchmarks/results/`. Colab Drive is not this repo.

The Colab badges below point at `main`. The matching notebooks are in
this checkout: `01_quickstart.ipynb`, `02_bandit_analysis.ipynb`,
`03_kernel_profiling.ipynb`, and `Colab_Benchmarks_T4.ipynb`.

Easily launch and explore FlashSpec notebooks in Google Colab:

| Notebook | Launch Badge | Description |
|----------|--------------|-------------|
| Quickstart | <a href="https://colab.research.google.com/github/Mattral/FlashSpec/blob/main/notebooks/01_quickstart.ipynb"><img alt="Quickstart" src="https://img.shields.io/badge/Open%20Quickstart-Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white"></a> | Introductory walkthrough of FlashSpec APIs and workflows |
| Bandit Analysis | <a href="https://colab.research.google.com/github/Mattral/FlashSpec/blob/main/notebooks/02_bandit_analysis.ipynb"><img alt="Bandit Analysis" src="https://img.shields.io/badge/Open%20Bandit%20Analysis-Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white"></a> | Contextual bandit experiments and evaluation pipelines |
| Kernel Profiling | <a href="https://colab.research.google.com/github/Mattral/FlashSpec/blob/main/notebooks/03_kernel_profiling.ipynb"><img alt="Kernel Profiling" src="https://img.shields.io/badge/Open%20Kernel%20Profiling-Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white"></a> | Profiling kernels and analyzing performance bottlenecks |
| Benchmarks T4 | <a href="https://colab.research.google.com/github/Mattral/FlashSpec/blob/main/notebooks/Colab_Benchmarks_T4.ipynb"><img alt="Benchmarks T4" src="https://img.shields.io/badge/Open%20Benchmarks%20T4-Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white"></a> | GPU benchmarks on NVIDIA T4, comparing throughput and latency |

---

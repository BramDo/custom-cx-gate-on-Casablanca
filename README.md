# Source code for: Qiskit Pulse: Programming Quantum Computers Through the Cloud with Pulses
This directory contains source code and data for constructing, running and analyzing the experiments described in the paper "Qiskit Pulse: Programming Quantum Computers Through the Cloud with Pulses" by Thomas Alexander et al.

The top level of this directory contains an `environment.yaml` file which may be used to create a virtual `conda` environment that is compatible with the code in the directory. Just run:
```shell
conda env create -f environment.yml -n qiskit-pulse-paper
```

Please see [here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) for more detail on Conda environments.

If you don't use Conda, you can use `requirements.txt` file in your environment. Just run:
```shell
pip install -r requirements.txt
```

The directory contains Jupyter notebooks for the experiments in the paper at the top-level, saved data from the experiments in the `data` folder, analysis and utility code in the `utils` folder and output figures in the `figs` folder.

To run experiments an IBM system credentials must [first be loaded]([https://github.com/Qiskit/qiskit-terra#executing-your-code-on-a-real-quantum-chip) and then a `hub`/`group`/`project` with access to a Qiskit pulse enabled backends used within the notebooks must be used to configure an IBMQ provider. For more help see the [Qiskit documentation](https://qiskit.org/documentation/).

# Notebooks
- `custom_cnot_experiments`: Notebook for constructing and calibrating CNOT gates using Qiskit Pulse an IBM Quantum computer. Is runnable without rerunning experiments by using cached data in the folder `data/custom_cnot_experiments`. Uses the backend `ibmq_almaden`.
- `cnot_paper_figures`: Code for reproducing cnot paper figures. Is runnable without rerunning experiments by using cached data in the folder `data/custom_cnot_experiments`.
- `qubit_state_discrimination`: Notebook for constructing and training qubit state discriminators and analyzing measurement correlation. Is runnable without rerunning experiments by using cached data in the folder `data/discriminator_efficiency_experiments`. Uses the backend `ibmq_singapore`.
- `paper_code_examples`: Example code for snippets contained within paper.

# Data
Results of experiments are stored using Python pickle. These can be read directly into Python and instantiated as objects if the environment is properly configured with the above `environment.yal`. Notebooks reuse data for analysis if `use_cache=True`, however, if this flag is set `False` all experiments will be re-run and old data will be overwritten with new. Rerunning all experiments may take some time depending on the `backend` selected.

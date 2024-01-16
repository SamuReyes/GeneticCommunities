# Genetic Communities

The aim of this practice is to detect communities in an Amazon item network, where communities are represented by the product category, and two products are connected if they have been repeatedly purchased together. In this paper, two methods for community detection will be compared: the Leiden algorithm and multi-objective genetic algorithms.

## 🚀 Install

To install the necessary dependencies for this project, ensure you have Python 3.11.4 or higher installed. Then, follow these steps:
1. Clone this repository to your local machine using:
```sh
$ git clone https://github.com/SamuReyes/GeneticCommunities
```
2. Change to the cloned repository directory:
```sh
$ cd GeneticCommunities
```
3. Install the dependencies using `pip`. It is recommended to do this within a virtual environment to avoid dependency conflicts. You can create a virtual environment using `venv`:
```sh
$ python -m venv venv
$ source venv/bin/activate # On Windows, use: venv\Scripts\activate
```
4. With the virtual environment activated, install the dependencies by running:
```sh
$ pip install -r requirements.txt
```
By following these steps, you will have all the necessary dependencies installed and be ready to run the project.

## ☕ Usage

This project is primarily interacted with through Jupyter Notebooks, which provide a user-friendly interface for running various functions and algorithms.

### Main Notebook: `main.ipynb`

The `main.ipynb` notebook is the central file where you can perform a variety of tasks:

- **Execute the Leiden Algorithm:** Run the Leiden algorithm for community detection.
- **Launch Genetic Algorithms:** Initiate genetic algorithms for obtaining communities distributions.
- **Compare Results with Ground Truth:** Analyze and compare the results of the algorithms with ground truth data.
- **Metric Evaluation:** Evaluate different metrics to measure the performance and effectiveness of the algorithms.

Each section within the notebook is well-documented, providing clear instructions and explanations on how to execute and utilize each function.

### Hyperparameter Tuning Notebook: `hp_tunning.ipynb`

In addition to the main notebook, there is a `hp_tunning.ipynb` notebook specifically designed for hyperparameter tuning of the genetic algorithms.

## ▶️ Getting Started

To start using these notebooks:

1. Ensure you have followed the Installation of Dependencies section to set up your environment.
2. Open the desired notebook (`main.ipynb` or `hp_tunning.ipynb`) in Jupyter Notebook or JupyterLab.
3. Follow the instructions and documentation within each notebook to execute different functions and algorithms.

These notebooks provide a comprehensive guide and hands-on approach to understanding and utilizing the capabilities of this project.



## 🤝 Authors

[Samuel Reyes Sanz](https://github.com/SamuReyes)

[Eduardo Riederer](https://github.com/emriederer)

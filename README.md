# Fitting hyperelasitc materials models to experimental data.
The purpose of this repository is to serve as an a gentle introduction to material parameters identification as well as for teaching works.
purposes

**Abstract**: Ogden–Hill model is the most wisely used material law for large strain simulations of compressible elastic materials using FE software. It is implemented in ABAQUS as "hyperfoam" model in which the users need to provide a set of material parameters for a specific specimen. These parameters can be indentified seperately on each experimental data or simutaniously in many of them using a non-linear curve fitting procedure, one instance of such experimental data are presented in [Kossa and Berezvai, 2016a], namely uniaxial compression (UA) and biaxial compression (BA). However, [Kossa and Berezvai, 2016b] only fitted a set of material parameters based on the UA dataset led poor performace of BA dataset with $R^2_{UA} = 0.99828$, $R^2_{BA} = 0.88141$ respectively. In this work, both of the dataset UA and BA are used for the curve fitting process to achieve a better set of parameters for the Ogden-Hill's model which are ready to use in ABAQUS with $R^2_{UA} = 0.99978$, $R^2_{BA} = 0.99921$ respectively. The implementation takes advantage of hardware accelerated, batchable and automatic differentiable optimizers using `JAXopt`, a library of `JAX`.

![plot](./ogden3_thiswork.png?raw=true "plot")

## Installation
Clone this repo
```bash
# Clone repo
git clone https://github.com/chauminhvu/fitting-model-parameters.git
```
Use [Docker](https://www.ibm.com/cloud/learn/docker) to create an image that contains all neccessary pakages for this repo: `docker -t <name of image> build .`
```bash
# change to repo's directory
cd ./fitting-model-parameters
# create docker image use with <name>
docker -t fit-model-params build .
```

Create container in interactive mode `-ti`, with name ("name of container"), and `-v` share working folder (e.g. $(pwd): share present directorry) with container, then run command `/bin/bash`, syntax: `docker run -ti --name <name of container> -v /local/folder:/container/folder <name of image> /bin/bash`

```bash
# run container
docker run -ti --name test-fiting -v $(pwd):/home/works fit-model-params bin/bash
# change dir. to works
cd /home/works
```
## Usage
Run non-linear optimisation to fit parameters of Ogden model:
```bash
python3 fitting-ogden-model.py
```
Files details:

`data:` folder contained data files.

`./data/UAexp_polyethylene_foam`: UA experimental data.

`./data/BAexp_polyethylene_foam`: BA experimental data.

`.gitignore`: specifies intentionally untracked files that Git should ignore. 

`.github/workflows/ci.yml`: [workflow](https://docs.github.com/en/actions/learn-github-actions/understanding-github-actions) of this repo.


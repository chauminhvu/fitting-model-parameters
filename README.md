# Fitting hyperelasitc materials models to experimental data.

![CI testing](https://github.com/chauminhvu/fitting-model-parameters/actions/workflows/ci.yml/badge.svg?barnch=main)

The purpose of this repository is to serve as an a gentle introduction to material parameters identification as well as for teaching purposes.

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
---
Tips:

If [Visual Studio Code](https://code.visualstudio.com/download) is used, it will work well with the [Docker extension](https://marketplace.visualstudio.com/items?itemName=ms-azuretools.vscode-docker)
([add it in VS code](https://youtu.be/5s6M4w7ucUI?t=17)),
after build the image and run the container, click on the "Docker extension" ([hint](https://youtu.be/GBl9CR8tlXk?t=172))/ right-click on the container and choose `Attach Visual Studio Code` -> start working.

---

<br>

## Usage
Run non-linear optimisation to fit parameters of Ogden model:
```bash
python3 fitting-ogden-model.py
```

<br>

## Files organisation:
<!--
$ git config --global alias.tree '! git ls-tree --full-name --name-only -t -r HEAD | sed -e "s/[^-][^\/]*\//   |/g" -e "s/|\([^ ]\)/|-- \1/"'
$ git tree
 -->

```bash
.github
   |-- workflows
   |   |-- ci.yml                   <- workflow
.gitignore    <- specifies intentionally untracked files that Git should ignore.
Dockerfile                          <- generate docker image
README.md                           <- The top-level README
data
   |-- BAexp_polyethylene_foam.csv  <- UA experimental data
   |-- UAexp_polyethylene_foam.csv  <- BA experimental data
fitting-ogden-model.py              <- Fit parameters

```
`ci.yml`: [workflow](https://docs.github.com/en/actions/learn-github-actions/understanding-github-actions) of this repo.

<br>

<!-- CONTRIBUTING -->
## Contributing
If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<br>

<!-- Authors -->
## Authors
Vu M. Chau - chauminhvu@gmail.com
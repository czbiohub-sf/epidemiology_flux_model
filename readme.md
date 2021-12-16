# Supplementary information

The repository contains supplementary information for the manuscript :
*A model based on a high-resolution flux matrix explains the spread of
diseases in a spatial network and the effect of mitigation strategies*.

Please start at the [index](notebooks/0-index.ipynb) notebook to be guided through
the results.

To be able to execute the Jupyter notebooks, Python 3.8.5 together with
specific libraries are required. For convenience, a Jupyter server can
be started locally using docker.

## Prerequisites

You'll need a working installation of docker (see the
[doc](https://docs.docker.com/get-docker/)). Before being
able to start a jupyter server through docker, you will need to create
an image.

First, open and edit the Dockerfile located at
`docker/python-nvidia/Dockerfile`. Change the values of the `UID` and
`GID` variables to match your own values. This can be checked by running the
command `ls -ln`. Save and close the file. Build the docker image by executing
(at the root of the repository):

```
docker build -t <yourname>/python-nvidia docker/python-nvidia/.
```

Note that this Docker image is based on `nvidia/cuda:11.0-devel`. If
this is not compatible with your system, you will need to use a
different Debian image (eg `ubuntu`), remove the `cupy-cuda110` library
in the `docker/python-nvidia/resources/requirements.txt` file, and
modify the scripts using CuPy (eg use `NumPy` instead).

## Running the notebooks in a Jupyter server Once the docker image is
Once the image has been built, edit the `start_jupyter_server.sh` and change the value of the
variable `dimage` to match the name of the docker image created in the
last step. If you are not using a GPU (see previous section) set the
value of the variable `gpus` to `""`. Execute in a terminal (at the root
of the repository):

```
bash start_jupyter_server.sh
```

Copy the prompted token and paste it in the corresponding field when
entering the jupyter server. The jupyter server can be accessed by
entering `http://localhost:8889` in a web browser.

You can also execute all notebooks at once using the convenience script provided:
```
bash execute_notebooks.sh
```

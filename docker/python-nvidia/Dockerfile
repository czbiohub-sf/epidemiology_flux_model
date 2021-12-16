FROM nvidia/cuda:11.0-devel
# FROM nvidia/cuda:11.0-runtime

# ports
# for jupyter, needs to be passed with -p xxxx:8888 at run time
EXPOSE 8888

########## preconfig ##########
USER root
ENV DEBIAN_FRONTEND=noninteractive
RUN export DEBIAN_FRONTEND

RUN apt update -y \
    && apt install -y \
    libssl-dev \
    python3-dev \
    python3-pip \
    && apt-get clean

RUN rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

RUN pip3 install --no-cache-dir install --upgrade pip

# Prepare environment
RUN mkdir /tp
WORKDIR /tp

# Install dependencies
COPY resources/apt.txt /tp/
RUN apt update && cat apt.txt | xargs apt install -y
RUN curl -sL https://deb.nodesource.com/setup_15.x | bash - && apt install -y nodejs

########## python ##########
COPY resources/requirements.txt /tp/
RUN pip install -r requirements.txt --no-cache-dir

########## jupyter lab ##########
RUN pip install jupyterlab "ipywidgets>=7.5"
RUN jupyter nbextension enable --py widgetsnbextension

# needed otherwise bug when opening terminal from jupyter.
ENV SHELL=/bin/bash

##########################
#add new sudo user
RUN apt install sudo
ENV USERNAME python
ENV GROUPNAME python
ENV UID 5190
ENV GID 5000
RUN groupadd --gid $GID $GROUPNAME && \
     useradd -g $GROUPNAME -u $UID -m $USERNAME && \
     echo "$USERNAME:$USERNAME" | chpasswd && \
     usermod --shell /bin/bash $USERNAME && \
     usermod -aG sudo $USERNAME && \
     echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/$USERNAME && \
     chmod 0440 /etc/sudoers.d/$USERNAME

USER $USERNAME

########## docker behavior ##########
#ENTRYPOINT ["python"]
CMD /bin/bash



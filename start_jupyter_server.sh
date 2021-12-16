dport=8889
duser=python
dpath=$PWD/docker/python-nvidia
# gpus=
gpus="--gpus all"
dimage=gletreut/python-nvidia
dname=python-nvidia
dcmd="jupyter lab --no-browser --ip 0.0.0.0"

docker run --rm \
  -it \
  -p $dport:8888  \
  -v $PWD:/home/$duser/shared \
  -v $dpath/resources/jupyter:/home/$duser/.jupyter \
  -v $dpath/resources/bashrc:/home/$duser/.bashrc \
  -w /home/$duser/shared \
  $gpus \
  -u $duser -h $dname --name $dname  \
  $dimage $dcmd

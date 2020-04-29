PROJECT_DIR="~/Workspace/cs590"
DATA_DIR="/usr/project/xtmp/fny/cs590"

cd $PROJECT_DIR

#
# Anaconda
#

wget https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh
sh Anaconda3-2020.02-Linux-x86_64.sh

#
# Protein Solver
#

wget https://github.com/ostrokach/proteinsolver/archive/v0.1.16.tar.gz -O proteinsolver.tar.gz
tar -xvf proteinsolver.tar.gz
cd proteinsolver

conda create -n proteinsolver -c pytorch -c conda-forge -c kimlab -c ostrokach-forge proteinsolver
conda activate proteinsolver

pip install --editable .

# Install pretrained models
wget -r -nH --cut-dirs 1 --reject "index.html*" "http://models.proteinsolver.org/v0.1/"

# Training and validation data
cd $DATA_DIR/proteinsolver
wget -r -nH --reject "index.html*" "http://deep-protein-gen.data.proteinsolver.org/"
cd $PROJECT_DIR

#
# ConFind
#

wget https://grigoryanlab.org/confind/confind-msl-bin.tar.gz
tar -xvf confind-msl-bin.tar.gz
#
# MASTER
#

wget https://grigoryanlab.org/master/master-bin-v1.6.tar.gz
tar -xvf master-bin-v1.6.tar.gz

#
# TERMs
#

mkdir -p $DATA_DIR/TERMs
rsync -vaRrz arteni.cs.dartmouth.edu::univ-TERMs/TERMs $DATA_DIR
cd $PROJECT_DIR

#!/bin/bash
#SBATCH --partition compsci-gpu
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --mem-per-cpu 8G
#SBATCH --time 1-0:00:00
#SBATCH --job-name jprotnotebook
#SBATCH --output log/jprotnotebook-%J.log

# get tunneling info
XDG_RUNTIME_DIR=""
port=$PORT #${PORT:$(shuf -i8000-9999 -n1)}
node=$(hostname -s)
user=$(whoami)
cluster=$(hostname -f | awk -F"." '{print $2}')

# print tunneling instructions jupyter-log
echo -e "
MacOS or linux terminal command to create your ssh tunnel
ssh -N -L ${port}:${node}:${port} ${user}@sbatch.cs.duke.edu

Use a Browser on your local machine to go to:
localhost:${port}  (prefix w/ https:// if using password)
"

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/home3/fny/.anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/home3/fny/.anaconda3/etc/profile.d/conda.sh" ]; then
        . "/home/home3/fny/.anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/home3/fny/.anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

conda activate $CENV
# DON'T USE ADDRESS BELOW.
# DO USE TOKEN BELOW
jupyter-notebook --no-browser --port=${port} --ip=${node} --ContentsManager.allow_hidden=True

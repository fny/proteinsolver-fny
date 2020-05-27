#!/bin/bash
#SBATCH --partition compsci
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --mem-per-cpu 8G
#SBATCH --time 1-0:00:00
#SBATCH --job-name scontact
#SBATCH --output log/scontact-%J.log

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

conda activate proteinsolver
cd ~/cs590/notebooks
python find_contacts.py

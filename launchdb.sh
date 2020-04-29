#!/bin/bash
#SBATCH --partition compsci
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --mem-per-cpu 8G
#SBATCH --time 1-0:00:00
#SBATCH --job-name postgres
#SBATCH --output log/postgres-%J.log

port=5432
host=$(hostname -s)
user=$(whoami)

echo -e "
Postgres is launching on host=$host port=$port
"

~/Applications/postgres/bin/postgres -i -p $port -D  ~/cs590/data/postgres -k ~/Applications/postgres/var 2>&1 | tee -a log/postgres.log

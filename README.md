# Explorations in Generating Structures from Sequences

## Getting Started

See `setup.sh` for details about binaries that need to be installed and external files. I do not recommend you run the file since certain commands (i.e. downloading the MASTER database) take a while. You should probably cut and paste.

If you are working from a machine with an x86_64 architecture, you should be good to go with the binaries in `./bin`. Otherwise, compile everything that's needed in `setup.sh`.

You will need to install the latest version of anaconda. Afterwards run `conda create env -f environment.yml` to build the conda environment for this project.

### Installing PostgreSQL

We use postgres to store data.

After you've compiled postgresql, move the entire folder somewhere you can access (i.e. `~/Applications/postgres`).
You'll now be able to access the binaries at `~/Applications/posrgres/bin`.

First, create a new database:

```
~/Applications/postgres/bin/initdb -D ~/data/cs590/postgres -U postgres
```

In `~/data/cs590/postgres` edit `pg_hba.conf` to allow all connections:

```
host    all             all             0.0.0.0/0            md5
```

Now edit `pg_hba.conf` to listent to all incoming connections:
```
listen_addresses = '*'
```

You should now be able to launch a database with `sh launchdb.sh` in the `cs590` folder.

Access the database `~/Applications/postgres/bin/psql -h localhost -p 5432 -U postgres` and a user and database for `cs590`.

```
CREATE USER cs590 WITH PASSWORD 'password';
CREATE DATABASE cs590 OWNER cs590;
```

You can now launch the database from any machine on the cluster with `sbatch launchdb.sh`. To access the server just specify the host and port `~/Applications/postgres/bin/psql -h linux15 -p 5432 -U cs590`

***!!! Please change the password unless you're fine with randos on the cluster potentially mucking with your data !!!***

### Additional Changes

Make sure you have a log directory in root.

```
mkdir -p log
```

Also make sure all data is in a `data` folder. This is currently symlinked to `/usr/project/xtmp/fny/cs590` in the project on the Duke cluster. Feel free to copy this over to your own scratch space.

Clone [ProteinSolver](https://ostrokach.gitlab.io/project/proteinsolver/) into the project directory. After downloading the pretrained models, move `protein_train` to the project root.

All binaries should be placed in a project level `bin` folder.

## Running ProteinSolver


```
# Start a GPU session
sh bash_gpu.sh

# Set the port to whatever you like and start jupyter
PORT=4321 CENV=proteinsolver sh jprotnotebook.sh

# Follow the instructions in the prompt given to open jupyter
# in your browser. Then open notebooks/proteinsolver.ipynb.
```

## Running the Markov Models


```
# Start a CPU session
sh bash_cpu.sh

# Launch the database
sbatch launchdb.sh

# Set the port to whatever you like and start jupyter
PORT=1234 CENV=cs590 sh jprotnotebook.sh

# Follow the instructions in the prompt given to open jupyter
# in your browser. Then open notebooks/markov.ipynb.
```

## Additional Details


### ConFind

[ConFind](https://grigoryanlab.org/confind/) is a tool for finding residues that are in contact. It is built on the [MSL library](http://msl-libraries.org/index.php/Main_Page).

Example usage:

```
./bin/confind --p 000000.pdb --rLib bin/rotlibs
```

Unfortunately, there is no clear explanation of the output of ConFind outside of the source code in `confind-msl/mslib/myProgs/gevorg/confind.cpp`. From what I gather the format is as follows:

```
percont OR contact, position_id_i, position_id_j, degree, name_i, name_j
```

Some additional information:

 - percont: permanent contact
 - contact: contact
 - sumcond: sum of contact degrees

More details can be found in the [original paper](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0178272#sec015) particularly in the section linked.


"Sequence statistics of tertiary structural motifs reflect protein stability", F. Zheng, G. Grigoryan, PLoS ONE, 12(5): e0178272, 2017.

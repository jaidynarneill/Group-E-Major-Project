
#!/bin/bash
#SBATCH --job-name=Na_DFT
#SBATCH --ntasks=16
#SBATCH --time=02:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

module use /projects/lh36/sdwi0002/opt/modulefiles
module load vasp/6.6.1

export I_MPI_HYDRA_BOOTSTRAP=slurm

mpirun -n "$SLURM_NTASKS" vasp_std
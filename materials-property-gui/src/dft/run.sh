#!/bin/bash
#SBATCH --job-name=Materials_DFT
#SBATCH --ntasks=32
#SBATCH --constraint=intel
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

source /etc/profile.d/modules.sh
module use /projects/lh36/sdwi0002/opt/modulefiles
module load vasp/6.6.1

export I_MPI_HYDRA_BOOTSTRAP=slurm

element=$(<.element)
cd "$SLURM_SUBMIT_DIR"
potgen "$element"
if [ ! -s POTCAR ]; then
	echo "potgen did not create POTCAR for $element" >&2
	exit 2
fi

enmax=$(awk '/ENMAX/ { for (i = 1; i <= NF; i++) if ($i == "ENMAX") { value = $(i + 2); gsub(/;/, "", value); print value; exit } }' POTCAR)
if [ -z "$enmax" ]; then
	echo "Could not read ENMAX from generated POTCAR for $element" >&2
	exit 2
fi
encut=$(awk -v value="$enmax" 'BEGIN { printf "%d", int(value * 1.3 + 0.999) }')

for task_directory in "$SLURM_SUBMIT_DIR"/tasks/*; do
	[ -d "$task_directory" ] || continue
	cp "$SLURM_SUBMIT_DIR/POTCAR" "$task_directory/POTCAR"
	sed "s/@ENCUT@/$encut/g" "$task_directory/INCAR.template" > "$task_directory/INCAR"
	printf 'Starting VASP task %s\n' "${task_directory##*/}"
	(
		cd "$task_directory"
		mpirun -n "$SLURM_NTASKS" vasp_std > vasp.log 2>&1
	)
done
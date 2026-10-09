
#!/bin/bash
#SBATCH --job-name=Materials_DFT
#SBATCH --ntasks=16
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

source /etc/profile.d/modules.sh
module use /projects/lh36/sdwi0002/opt/modulefiles
module load hpcx-ompi
module load hpcx/.2.14-redhat9.2-patch1
module load hdf5/1.12.3
module load wannier90/3.1.0-mpi
module load vasp/6.4.2

export I_MPI_HYDRA_BOOTSTRAP=slurm

: "${VASP_PP_PATH:?VASP_PP_PATH is not set by the VASP module}"
element=$(<.element)
potcar_source=''
for candidate in \
	"$VASP_PP_PATH"/potpaw_PBE*/"$element"/POTCAR \
	"$VASP_PP_PATH"/potpaw_PBE*/"$element"*/POTCAR \
	"$VASP_PP_PATH"/"$element"/POTCAR \
	"$VASP_PP_PATH"/"$element"*/POTCAR; do
	if [ -f "$candidate" ]; then
		potcar_source=$candidate
		break
	fi
done

if [ -z "$potcar_source" ]; then
	echo "No PBE POTCAR found for $element under $VASP_PP_PATH" >&2
	exit 2
fi

enmax=$(awk '/ENMAX/ { for (i = 1; i <= NF; i++) if ($i == "ENMAX") { value = $(i + 2); gsub(/;/, "", value); print value; exit } }' "$potcar_source")
if [ -z "$enmax" ]; then
	echo "Could not read ENMAX from $potcar_source" >&2
	exit 2
fi
encut=$(awk -v value="$enmax" 'BEGIN { printf "%d", int(value * 1.3 + 0.999) }')

for task_directory in "$SLURM_SUBMIT_DIR"/tasks/*; do
	[ -d "$task_directory" ] || continue
	cp "$potcar_source" "$task_directory/POTCAR"
	sed "s/@ENCUT@/$encut/g" "$task_directory/INCAR.template" > "$task_directory/INCAR"
	(
		cd "$task_directory"
		mpirun -n "$SLURM_NTASKS" vasp_std > vasp.log 2>&1
	)
done
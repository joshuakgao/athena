#!/bin/bash

INPUT_DIR="datasets/chessbench/data/train"
OUTPUT_DIR="datasets/chessbench/data_mate/train"
SCRIPT_PATH="datasets/chessbench/add_mating_data.py"
ENV_PATH="/project/hoskere/jkgao/.conda/envs/athena"
N_JOBS=401

mkdir -p "$OUTPUT_DIR"
mkdir -p slurm_jobs
mkdir -p logs

# --- Submit test bag file first ---
TEST_INPUT="datasets/chessbench/data/test/action_value_data.bag"
TEST_OUTPUT="datasets/chessbench/data_mate/test/action_value_data.bag"
TEST_JOB_SCRIPT="slurm_jobs/job_test_action_value.sh"

mkdir -p "$(dirname "$TEST_OUTPUT")"

cat <<EOF > "$TEST_JOB_SCRIPT"
#!/bin/bash
#SBATCH -J test
#SBATCH -o logs/test.out
#SBATCH --cpus-per-task=1
#SBATCH -t 10:0:0
#SBATCH --mem-per-cpu=4GB

module add Miniforge3/py3.10
module add cudatoolkit/12.4
source activate $ENV_PATH

export PYTHONPATH=$(pwd)

$ENV_PATH/bin/python $SCRIPT_PATH --input_bag="$TEST_INPUT" --output_bag="$TEST_OUTPUT"
EOF

sbatch "$TEST_JOB_SCRIPT"

# --- Now split training jobs into N_JOBS chunks ---
bag_files=("$INPUT_DIR"/*.bag)
total_files=${#bag_files[@]}
files_per_job=$(( (total_files + N_JOBS - 1) / N_JOBS ))

for ((i=0; i<N_JOBS; i++)); do
    start=$((i * files_per_job))
    end=$((start + files_per_job - 1))
    end=$((end < total_files ? end : total_files - 1))

    JOB_SCRIPT="slurm_jobs/job_chunk_${i}.sh"
    LOG_FILE="logs/chunk_${i}.out"

    cat <<EOF > "$JOB_SCRIPT"
#!/bin/bash
#SBATCH -J chunk${i}
#SBATCH -o $LOG_FILE
#SBATCH --cpus-per-task=1
#SBATCH -t 10:0:0
#SBATCH --mem-per-cpu=4GB

module add Miniforge3/py3.10
module add cudatoolkit/12.4
source activate $ENV_PATH

export PYTHONPATH=$(pwd)

EOF

    for ((j=start; j<=end; j++)); do
        input_file="${bag_files[j]}"
        base_name=$(basename "$input_file")
        output_file="$OUTPUT_DIR/$base_name"

        echo "$ENV_PATH/bin/python $SCRIPT_PATH --input_bag=\"$input_file\" --output_bag=\"$output_file\"" >> "$JOB_SCRIPT"
    done

    sbatch "$JOB_SCRIPT"
done

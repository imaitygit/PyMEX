#!/bin/bash
# submit_all.sh - run this once, it chains all jobs automatically
# Most efficient way of doing it

JOB1=$(sbatch --parsable job1_write_H.sh)
echo "Submitted Step 1 (write_H):     Job ID = $JOB1"

JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 job2_diagon.sh)
echo "Submitted Step 2 (diagon):      Job ID = $JOB2 (waits for $JOB1)"

JOB3=$(sbatch --parsable --dependency=afterok:$JOB2 job3_optical.sh)
echo "Submitted Step 3 (optical):     Job ID = $JOB3 (waits for $JOB2)"

echo ""
echo "Chain: $JOB1 -> $JOB2 -> $JOB3"
echo "Monitor: squeue -u imli"
echo "Cancel all: scancel $JOB1 $JOB2 $JOB3"

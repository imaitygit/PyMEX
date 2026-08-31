#!/bin/bash
set -euo pipefail
SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# optional: depend first job on previous chain's last job
DEP=""
if [ -n "${PREV_JID:-}" ]; then DEP="--dependency=afterok:${PREV_JID}"; fi
JID1=$(sbatch --parsable $DEP "${SCRIPTS_DIR}/scf.sh")
echo "Submitted SCF           → ${JID1}"

JID2=$(sbatch --parsable --dependency=afterok:${JID1} "${SCRIPTS_DIR}/nscf.sh")
echo "Submitted NSCF          → ${JID2}  (after ${JID1})"

JID3=$(sbatch --parsable --dependency=afterok:${JID2} "${SCRIPTS_DIR}/wannier_pp.sh")
echo "Submitted Wannier90-pp  → ${JID3}  (after ${JID2})"

JID4=$(sbatch --parsable --dependency=afterok:${JID3} "${SCRIPTS_DIR}/pw2wan.sh")
echo "Submitted PW2Wannier90  → ${JID4}  (after ${JID3})"

JID5=$(sbatch --parsable --dependency=afterok:${JID4} "${SCRIPTS_DIR}/wannier.sh")
echo "Submitted Wannier90     → ${JID5}  (after ${JID4})"

echo "Chain: ${JID1} → ${JID2} → ${JID3} → ${JID4} → ${JID5}"
echo ${JID5}

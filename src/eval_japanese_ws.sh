#!/bin/sh
# $ bash  eval_japanese_ws.sh sys_file ref_file
sys_file=$2
ref_file=$1

echo "ref_file : ${ref_file}"
echo "sys_file : ${sys_file}"

python pre_treatment.py ${ref_file} ${sys_file} |perl conlleval.pl -d "\t"


#! /bin/bash

pipeline_dir=$1
sub_task=$2
log_dir=$3
megatron_dir=$pipeline_dir/Megatron-LM

cd $megatron_dir

# Enable adaptor
for file_dir in $megatron_dir/tests_extend/system_tests/${sub_task}/*;
do
    file_name=${file_dir##*/}
    task_name="${file_name%.*}"
    sed -i '60a\    --exit-interval 100 \\' tests_extend/system_tests/${sub_task}/${task_name}.sh

    echo "------------------ Task $task_name Start ------------------"
    echo "====================$task_name====================" >> ${log_dir}/result.log
    echo "[Start time] $(date "+%Y/%m/%d %H:%M:%S")" >> ${log_dir}/result.log
    run_cmd="bash tests_extend/system_tests/${sub_task}/${task_name}.sh 2>&1 | tee ${log_dir}/${task_name}.log"
    rm -rf ckpt_llama
    eval ${run_cmd}
    
    # TPS
    step_time=`grep -a "elapsed time per iteration" ${log_dir}/${task_name}.log|awk -F "|" '{print $3}'|awk -F ":" '{print $2}'|tail -n -100|awk '{sum+=$1} END {print"",sum/NR}'`
    global_batch_size=`grep -w "global_batch_size" ${log_dir}/${task_name}.log|awk -F " " '{print $3}'|tail -n -1`
    seq_length=`grep -w "seq_length" ${log_dir}/${task_name}.log|awk -F " " '{print $3}'|tail -n -1`
    world_size=`grep -w "world_size" ${log_dir}/${task_name}.log|awk -F " " '{print $3}'|tail -n -1`
    TPS=$(echo "$global_batch_size * $seq_length * 1000 / $world_size / $step_time"|bc)
    
    # Loss
    last_loss=`grep "elapsed time per iteration" ${log_dir}/${task_name}.log|awk -F "loss:" '{print $2}'|awk -F " " '{print $1}'|tail -n -1`

    # Gather the results
    echo "[ActualFPS] ${TPS}" >> ${log_dir}/result.log
    echo "[ActualLoss] ${last_loss}" >> ${log_dir}/result.log
    echo -e "[End time] $(date "+%Y/%m/%d %H:%M:%S")\n" >> ${log_dir}/result.log
    echo -e "------------------ Task $task_name Done ------------------\n"
done

set +x

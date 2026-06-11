#/bin/bash

current_dir=$(dirname $(readlink -f $0))
pipeline_dir=$(dirname $(dirname $current_dir))
megatron_dir=$pipeline_dir/Megatron-LM
cann_dir=0
baseline_dir=0
DATESTR=$(date +"%m-%d")"-"$(date +"%H-%M")
log_dir=$pipeline_dir/logs/$DATESTR
mkdir -p $log_dir
mkdir $pipeline_dir/old_logs

#Check Parameters
for para in $*
do
    if [[ $para == --cann_dir* ]];then
        cann_dir=`echo ${para#*=}`
    fi
    if [[ $para == --baseline_dir* ]];then
        baseline_dir=`echo ${para#*=}`
    fi
done

#Check Megatron-LM
if [ ! -d $megatron_dir ];
then
    echo "> Please confirm that Megatron-LM and MindSpeed are in the same folder!!!"
    exit 0
else
    echo "> Megatron-LM exists in ${pipeline_dir}."
    rm -rf $megatron_dir/tests_extend
    cp -r $pipeline_dir/MindSpeed/tests_extend $megatron_dir
fi

#Source CANN
if [ -d $cann_dir ];
then
    echo "> Sourcing ${cann_dir}/ascend-toolkit/set_env.sh."
    test ! -d $cann_dir/ascend-toolkit && echo "Error: Ascend-toolkit is not installed in ${cann_dir}!!!" && exit 0
    source $cann_dir/ascend-toolkit/set_env.sh
fi

#Check Python
echo "> Use local env......"
for sub_task in feature_tests;
do
    bash ${current_dir}/test_pipeline.sh ${pipeline_dir} ${sub_task} ${log_dir}
done

echo "> Analyzing logs......"
if [ ! -d $baseline_dir ];
then
    python ${current_dir}/logpost_ci.py --ci-path=${log_dir}
else
    python ${current_dir}/logpost_ci.py --ci-path=${log_dir} --baseline-path=${baseline_dir} --cidate=$DATESTR
fi

cd $pipeline_dir/logs
mv *.tar.gz ../old_logs
tar -cvf ${DATESTR}.tar.gz ${DATESTR}
export GLOO_SOCKET_IFNAME=eth0
export WANDB_MODE=disabled

MAXLEN=8192
EPOCH=3
SAVEINTERVAL=${EPOCH}
LR=1e-6
BETA=0.1
DPOP_LAMBADA=50.0
SFT_LAMBADA=1.0
LenN_LAMBADA=0.01
TDPO_ALPHA=0.5
SIMPO_LAMBADA=0.3
LR_TYPE=linear

LNORM=$1
RLType=$2  # ipo or sigmoid (dpo) or kto_pair or dpop or mix or lenN or orpo or tdpo or bco_pair or sppo_hard or simpo or sampo or nca

raw_model_path=llama3-8b-instruct/
train_data_path=dummy.jsonl
eval_data_path=dummy.jsonl
deepspeed_config_path=ds_config.json
model_output_path=${RLType}_$(basename ${raw_model_path})/

if [ ${RLType} == "simpo" ]
then
    BETA=2.5
    EPOCH=1
fi

case ${raw_model_path} in 
    *"2.8b"*)
        PER_GPU_BATCH=4
        GRA_ACC=4
        MAXLEN=1024
        EPOCH=1
        BETA=0.5
        ;;
    *"8b"*)
        PER_GPU_BATCH=1
        GRA_ACC=16
        ;;
    *"13b"*)
        PER_GPU_BATCH=1
        GRA_ACC=16
        ;;
    *"llama3"*)
        LR=4e-7
        SIMPO_LAMBADA=1.4
        ;;
esac

TRAINDATANUM=$(wc -l < "${train_data_path}")
SAVESTEP=$(awk "BEGIN {print int(${TRAINDATANUM} * ${EPOCH} / (${PER_GPU_BATCH} * ${GRA_ACC} * $GPU_NUM_PER_NODE * ${SAVEINTERVAL} * $NODE_NUM)) + 1}")
TOTALSTEP=$(awk "BEGIN {print int(${TRAINDATANUM} * ${EPOCH} / (${PER_GPU_BATCH} * ${GRA_ACC} * $GPU_NUM_PER_NODE * $NODE_NUM)) + 1}")
EVALSTEP=100

echo "We use $NODE_NUM nodes to train with ${TRAINDATANUM} samples for ${EPOCH} epochs, resulting in ${TOTALSTEP} running steps, and thus we will save checkpoints every ${SAVESTEP} steps."

# training
torchrun --nnodes=$NODE_NUM \
    --node_rank=$INDEX \
    --nproc_per_node $GPU_NUM_PER_NODE \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    dpo_tuning_demo.py \
    --model_name_or_path ${raw_model_path} \
    --bf16 True \
    --output_dir ${model_output_path} \
    --num_train_epochs ${EPOCH} \
    --per_device_train_batch_size ${PER_GPU_BATCH} \
    --gradient_accumulation_steps ${GRA_ACC} \
    --save_strategy "steps" \
    --save_steps ${SAVESTEP} \
    --save_total_limit 5 \
    --per_device_eval_batch_size ${PER_GPU_BATCH} \
    --evaluation_strategy "steps" \
    --eval_steps ${EVALSTEP} \
    --learning_rate ${LR} \
    --log_level "info" \
    --logging_strategy "steps" \
    --logging_steps 1 \
    --weight_decay 0.0 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type ${LR_TYPE} \
    --deepspeed ${deepspeed_config_path} \
    --tf32 True \
    --model_max_length ${MAXLEN} \
    --train_data_path ${train_data_path} \
    --eval_data_path ${eval_data_path} \
    --preprocessing_num_workers 32 \
    --dataloader_num_workers 32 \
    --gradient_checkpointing True \
    --report_to "none" \
    --loss_type ${RLType} \
    --dpo_beta ${BETA} \
    --len_norm ${LNORM} \
    --dpop_lambda ${DPOP_LAMBADA} \
    --sft_lambda ${SFT_LAMBADA} \
    --lenN_lambda ${LenN_LAMBADA} \
    --tdpo_alpha ${TDPO_ALPHA} \
    --simpo_lambda ${SIMPO_LAMBADA}

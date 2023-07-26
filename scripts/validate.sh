set -x

CONFIG=$1
CKPT=$2
PORT=2333

HOST=$(hostname -i)

python ./scripts/validate.py \
    --cfg ${CONFIG} \
    --valid-batch 1 \
    --checkpoint ${CKPT} \
    --launcher pytorch --rank 0 \
    --dist-url tcp://${HOST}:${PORT} \

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Emulated node 0 (master) on a single 4-GPU instance: uses GPUs 0 and 1.
# On two real machines, run the same torchrun command on each machine with
# --nproc_per_node=4 and without CUDA_VISIBLE_DEVICES / NCCL_HOSTID.

MasterIPaddress=$(hostname -I | awk '{print $1}')
echo "Master IP Address: " $MasterIPaddress

export CUDA_VISIBLE_DEVICES=0,1      # GPUs that belong to this node
if [ "${EMULATE_NETWORK:-1}" = "1" ]; then
    export NCCL_HOSTID=node0         # NCCL treats this node as a separate host -> network transport
fi
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

cd "$(dirname "$0")/../.."           # workspace directory
torchrun --nproc_per_node=2 --nnodes=2 --node_rank=0 --master_addr=$MasterIPaddress --master_port=1234 \
    source_code/test_ddp.py --num_epochs ${NUM_EPOCHS:-2} --model_filename resnet_node0.pth

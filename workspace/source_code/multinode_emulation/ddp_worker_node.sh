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

# Emulated node 1 (worker) on a single 4-GPU instance: uses GPUs 2 and 3.
# Usage: ddp_worker_node.sh <master IP address>

MasterIPaddress=$1
if [ -z "$MasterIPaddress" ]; then
    echo "Usage: $0 <master IP address printed by ddp_master_node.sh>"
    exit 1
fi

export CUDA_VISIBLE_DEVICES=2,3      # GPUs that belong to this node
if [ "${EMULATE_NETWORK:-1}" = "1" ]; then
    export NCCL_HOSTID=node1         # NCCL treats this node as a separate host -> network transport
fi
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

cd "$(dirname "$0")/../.."           # workspace directory
torchrun --nproc_per_node=2 --nnodes=2 --node_rank=1 --master_addr=$MasterIPaddress --master_port=1234 \
    source_code/test_ddp.py --num_epochs ${NUM_EPOCHS:-2} --model_filename resnet_node1.pth

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

"""Adapt the Slurm-based lab notebooks to a single Brev VM with 4 GPUs.

- Strips `srun ...` and `module load ... &&` prefixes from code cells so
  commands run directly on the instance.
- Marks the multi-node and Transformer Engine (FP8) notebooks as unavailable.
- Points notebooks at the bootcamp Jupyter kernel registered by setup.sh.

Safe to run more than once. Usage: python adapt_notebooks.py <workspace_dir>
"""

import json
import re
import sys
from pathlib import Path

KERNEL_NAME = "profiling-bootcamp"
KERNEL_DISPLAY = "Profiling Bootcamp (Python 3.12)"
MARKER = "<!-- brev-adapted -->"

# Notebooks that run on a single node with 4 GPUs
LAB_NOTEBOOKS = [
    "system-topology",
    "data-parallelism",
    "model-parallelism",
    "nsys-introduction",
    "nsys-application",
    "nsys-trace",
    "nsight_advanced",
]

MODULE_LOAD = re.compile(r"module load \S+\s*&&\s*")
# e.g. "srun -p primary -N 1 --gres=gpu:4 " / "srun --partition=primary -n2 --gres=gpu:8 "
SRUN = re.compile(r"srun\s+(?:(?:--partition=\S+|-p\s+\S+|-N\s*\d+|-n\s*\d+|--gres=\S+)\s+)+")

NOT_AVAILABLE = " *(not available on this Brev environment)*"

BREV_NOTE = (
    MARKER + "\n"
    "> **Brev environment:** you are running on a single cloud instance with 4x NVIDIA A100 GPUs. "
    "Commands that originally used Slurm (`srun`) now run directly on the instance. "
    "Saved outputs and screenshots in the notebooks come from a DGX H100 cluster, "
    "so your GPU names, topology (`nvidia-smi topo -m`), and timings will differ. "
    "A100 GPUs do not support FP8."
)


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save(path, nb):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")


def source(cell):
    return "".join(cell["source"])


def set_source(cell, text):
    cell["source"] = text.splitlines(keepends=True)


def set_kernel(nb):
    nb.setdefault("metadata", {})["kernelspec"] = {
        "display_name": KERNEL_DISPLAY,
        "language": "python",
        "name": KERNEL_NAME,
    }


def has_marker(nb):
    return any(MARKER in source(c) for c in nb["cells"])


def note_cell():
    return {"cell_type": "markdown", "metadata": {}, "source": BREV_NOTE.splitlines(keepends=True)}


def adapt_lab(path):
    nb = load(path)
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        text = MODULE_LOAD.sub("", source(cell))
        text = SRUN.sub("", text)
        set_source(cell, text)
        leftover = re.search(r"\b(srun|sbatch|module load)\b", text)
        if leftover:
            raise SystemExit(f"{path.name}: could not adapt code cell:\n{text}")

    if path.stem == "system-topology" and not has_marker(nb):
        nb["cells"].insert(1, note_cell())

    # Lab 3 ends here on Brev; the original next notebook is multi-node
    for cell in nb["cells"]:
        if cell["cell_type"] == "markdown" and "(multinode.ipynb)" in source(cell):
            set_source(cell, source(cell).replace("[Next Notebook](multinode.ipynb)",
                                                  "[Back to Start](../start_here.ipynb)"))

    set_kernel(nb)
    save(path, nb)


def adapt_start_here(path):
    nb = load(path)
    for cell in nb["cells"]:
        text = source(cell)
        if "### Table of Contents" not in text or NOT_AVAILABLE in text:
            continue
        for link in ["(jupyter_notebook/multinode.ipynb)",
                     "(jupyter_notebook/transEng.ipynb)",
                     "(jupyter_notebook/nsys-fp8.ipynb)"]:
            text = text.replace(link, link + NOT_AVAILABLE)
        text = text.replace("- Lab 4: Transformer Engine", "- Lab 4: Transformer Engine" + NOT_AVAILABLE)
        set_source(cell, text)

    if not has_marker(nb):
        nb["cells"].insert(2, note_cell())
    set_kernel(nb)
    save(path, nb)


def main():
    workspace = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    adapt_start_here(workspace / "start_here.ipynb")
    for name in LAB_NOTEBOOKS:
        adapt_lab(workspace / "jupyter_notebook" / f"{name}.ipynb")
    print(f"Adapted notebooks in {workspace}")


if __name__ == "__main__":
    main()

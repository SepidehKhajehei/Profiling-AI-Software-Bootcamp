# Deploying the Bootcamp on NVIDIA Brev

This guide deploys the Profiling AI Software Bootcamp on [NVIDIA Brev](https://brev.nvidia.com). Each participant gets their own cloud instance with **4x NVIDIA A100 GPUs** (single node).

## Labs Included

| Lab | Notebook | On Brev (4x A100) |
|-----|----------|-------------------|
| Lab 1: System Topology | `system-topology.ipynb` | ✅ Yes |
| Lab 2: Distributed Training Strategy | `data-parallelism.ipynb` | ✅ Yes. The multi-node activities run on two emulated 2-GPU nodes on the same instance. |
| | `model-parallelism.ipynb` | ✅ Yes (uses 3 GPUs) |
| Lab 3: Performance Overview | `nsys-introduction.ipynb` | ✅ Yes |
| | `nsys-application.ipynb` | ✅ Yes |
| | `nsys-trace.ipynb` | ✅ Yes |
| | `nsight_advanced.ipynb` | ✅ Yes |
| | `multinode.ipynb` | ❌ No: needs 2 nodes and Slurm |
| Lab 4: Transformer Engine | `transEng.ipynb`, `nsys-fp8.ipynb` | ❌ No: needs FP8 (Hopper or newer) and Transformer Engine |

Expect about 3.5 hours for Labs 1–3.

## What the Brev Setup Does

The original notebooks run on a Slurm cluster. [brev/setup.sh](brev/setup.sh) prepares a Brev VM so the same material runs on a single instance:

1. Installs the latest Nsight Systems CLI from the NVIDIA CUDA apt repository.
2. Lets non-root users read GPU performance counters, which `nsys --gpu-metrics-devices` needs.
3. Creates the Python 3.12 `uv` environment from [requirements.txt](requirements.txt), as in [Deployment_Guide.md](Deployment_Guide.md).
4. Downloads the CIFAR-10 data and saved models. It also downloads MNIST ahead of time, because otherwise all 4 ranks try to download it at once.
5. Runs [brev/adapt_notebooks.py](brev/adapt_notebooks.py). This removes the `srun ...` and `module load ...` prefixes from code cells, marks the multi-node and Lab 4 notebooks as unavailable, and adds a note saying the saved outputs came from a DGX H100.
6. Registers a `Profiling Bootcamp (Python 3.12)` Jupyter kernel and starts JupyterLab as a systemd service on port `8888`, so it restarts after a reboot.

The notebooks are adapted on each instance, so the repository itself keeps working on Slurm clusters. The log is at `~/brev-bootcamp-setup.log`.

## Step 1: Publish the `brev/` Folder

The Launchable clones a **public** git repository, and that repository must contain the `brev/` folder. Either merge these files into the upstream repository, or push them to a public fork. Note the repository URL and branch.

## Step 2: Create the Launchable

In the [Brev console](https://brev.nvidia.com), open **Launchables → Create Launchable** and fill in the builder sections:

1. **Details**: name it, for example `Profiling AI Software Bootcamp`.
2. **Default hardware configuration**:
   - GPU: **4x NVIDIA A100** (80 GB or 40 GB both work).
   - Prefer an **SXM4 (NVLink)** instance type. On PCIe A100s, `nvidia-smi topo -p2p n` reports no NVLink, and the NVLink trace in `nsys-trace.ipynb` stays empty.
   - Disk: 200 GiB.
3. **Software configuration**: **VM Mode**.
   - Turn off Brev's built-in Jupyter, because the setup script runs its own JupyterLab on port 8888.
   - Paste this setup script, replacing the URL and branch with yours:

     ```bash
     #!/bin/bash
     export REPO_URL=https://github.com/openhackathons-org/Profiling-AI-Software-Bootcamp.git
     export REPO_BRANCH=main
     curl -fsSL "${REPO_URL%.git}/raw/${REPO_BRANCH}/brev/setup.sh" -o /tmp/bootcamp-setup.sh
     bash /tmp/bootcamp-setup.sh
     ```

     The script finds the repository if Brev has already cloned it. Otherwise it clones `REPO_URL` into `/home/ubuntu`.
4. **Source**: *I have code files in a git repository*, using the same public repository URL.
5. **Network**: add a **Secure Link** named `jupyter` on port **8888**. Secure Links require Brev sign-in, so JupyterLab runs without a token, bound to `127.0.0.1`.
6. **Launch parameters** (optional, all passed to the setup script as environment variables):
   - `DATA_ZIP_URL` and `MODELS_ZIP_URL`: mirror URLs for `data-list.zip` and `saved_models.zip`. Recommended for groups. See [Troubleshooting](#troubleshooting).
   - `JUPYTER_PORT`, `JUPYTER_IP`, `JUPYTER_TOKEN`: override the JupyterLab defaults.
7. **View access**: *Anyone with the link*, or *Only my organization*.
8. Click **Create Launchable** and copy the link.

## Step 3: Test One Instance Before the Event

1. Deploy one instance from the Launchable link and wait for setup to finish. It takes roughly 10–15 minutes.
2. Open a shell (`brev shell <instance-name>`) and check the end of the log:

   ```bash
   tail -n 30 ~/brev-bootcamp-setup.log
   ```

   It should list 4 GPUs, print an `nsys` version, report `cuda devices: 4`, and print no `WARNING` lines.
3. If the log says `GPU counters are still admin-only`, run `sudo reboot` once. JupyterLab starts again by itself after the reboot.
4. Check that profiling works:

   ```bash
   cd ~/Profiling-AI-Software-Bootcamp
   .venv/bin/nsys profile --gpu-metrics-devices=help
   grep RmProfilingAdminOnly /proc/driver/nvidia/params   # should be 0
   ```

5. Open the `jupyter` Secure Link from the instance's **Access** tab. Open `start_here.ipynb` and run every cell in Labs 1–3 at least once.

## Step 4: Give Participants Access

- **Self-service:** share the Launchable link. Each participant clicks **Deploy**, waits for setup, opens the `jupyter` Secure Link, and starts with `start_here.ipynb`. Each deployment is a separate 4-GPU instance billed to the organization that deploys it, so remind people to **stop or delete** their instance afterwards.
- **Pre-provisioned:** create the instances in advance and hand out links, for example:

  ```bash
  brev create bootcamp --gpu-name A100 --count <N> --startup-script @brev/setup.sh
  ```

  Brev's CLI flags change over time, so check `brev create --help` first.
  - When you pre-provision from the CLI, set `REPO_URL` and `REPO_BRANCH` inside the startup script.
  - Add the port 8888 tunnel on each instance from its **Access** tab.

Participants view `.nsys-rep` reports in the **Nsight Systems GUI** on their own computer. Download the reports from `workspace/reports/` in the JupyterLab file browser.

## Troubleshooting

- **Google Drive download fails (`Too many users have viewed or downloaded this file`).** Many instances downloading at once exceed Drive's quota. Host `data-list.zip` and `saved_models.zip` somewhere else, such as cloud object storage or a GitHub release, and set `DATA_ZIP_URL` and `MODELS_ZIP_URL`. To retry on an instance:
  `cd ~/Profiling-AI-Software-Bootcamp/workspace && ../.venv/bin/python source_code/download-data.py`, then run the two `unzip` commands from [Deployment_Guide.md](Deployment_Guide.md).
- **`ERR_NVGPUCTRPERM` or "insufficient privilege" when profiling with GPU metrics.** Run `sudo reboot` once so the `NVreg_RestrictProfilingToAdminUsers=0` setting takes effect.
- **Port 8888 already in use.** Brev's built-in Jupyter is running. Either turn it off in the Launchable, or keep it and choose the **Profiling Bootcamp (Python 3.12)** kernel, which the adapted notebooks already select.
- **Secure Link shows a blank page or cannot connect.** Check the service with `systemctl status bootcamp-jupyter`. If your provider's tunnel cannot reach `127.0.0.1`, set `JUPYTER_IP=0.0.0.0` together with a `JUPYTER_TOKEN`, and run the setup script again.
- **Re-running setup.** `brev/setup.sh` can safely be run again. It skips work that is already done.

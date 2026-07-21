# MUSIC: Learning Muscle-Driven Dexterous Hand Control

This is the official implementation of _MUSIC: Learning Muscle-Driven
Dexterous Hand Control_.
[[Webpage](https://pei-xu.github.io/music)]
[[arXiv](https://arxiv.org/abs/2604.23886)]
[[YouTube](https://youtu.be/tt1D36gE8zM)]
[[TOG](https://doi.org/10.1145/3811402)]

![MUSIC teaser](doc/teaser.png)

MUSIC is a hierarchical control framework for physics-based piano performance
with anatomically detailed musculoskeletal hands. General single-hand policies
first learn to track a large motion dataset at 480 Hz using muscle-tendon
actuation. Their motion spaces are distilled into compact variational latent
spaces, and piece-specific high-level policies coordinate both hands at 60 Hz
by controlling these latent actions from musical note goals.

## Code Usage

### Dependencies

- Python 3.13
- PyTorch 2.6.0
- JAX 0.4.34
- MuJoCo and MuJoCo MJX 3.3.5

We recommend using [uv](https://docs.astral.sh/uv/) to install the exact
dependencies:

```bash
uv sync
```

The supplied `pyproject.toml` installs the CUDA 12 build of JAX on Linux and
the standard JAX build on other platforms. Alternatively, create a Python 3.13
environment and install the requirements with pip:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

All commands below use `uv run`. On macOS, use `uv run mjpython main.py ...`
instead of `uv run python main.py ...` for evaluation commands that open the
MuJoCo viewer.

### Data Preparation

Download the 651-minute reference motion dataset from
[Hugging Face](https://huggingface.co/datasets/xupei0610/MUSIC):

```bash
uvx --from huggingface-hub hf download xupei0610/MUSIC \
  --repo-type dataset \
  --local-dir motions
```

The resulting directory should have the following layout:

```text
motions/
  left_hand_motions/
  right_hand_motions/
```

The configuration files enumerate the motion files when they are loaded, so
the dataset is required for both training and evaluation. On systems with a
low open-file limit, run the following command before starting:

```bash
ulimit -n 4096
```

Piano goals are specified with note-event files. See
[`notes/README.md`](notes/README.md) for the file format and fingering
conventions.

### Joint-Driven Piano Control

To train a piece-specific joint-driven policy:

```bash
uv run python main.py cfg/joint.py \
  --note notes/017-1_fingering.txt \
  --ckpt <checkpoint_directory>
```

We provide policies trained with and without annotated fingering in
`pretrained/joint`. Evaluate them with:

```bash
uv run python main.py cfg/joint.py \
  --note notes/017-1_fingering.txt \
  --ckpt pretrained/joint/joint_017-fingering \
  --test

uv run python main.py cfg/joint.py \
  --note notes/017-1_nofingering.txt \
  --ckpt pretrained/joint/joint_017-nofingering \
  --test
```

### Low-Level Muscle Tracking

The muscle-driven pipeline first trains one general motion-tracking policy for
each hand:

```bash
uv run python main.py cfg/muscle_tracking_left.py \
  --ckpt <left_checkpoint_directory>

uv run python main.py cfg/muscle_tracking_right.py \
  --ckpt <right_checkpoint_directory>
```

Pretrained tracking policies are provided in `pretrained/muscle_tracking`.
Evaluate them with:

```bash
uv run python main.py cfg/muscle_tracking_left.py \
  --ckpt pretrained/muscle_tracking/muscle_tracking_left \
  --test

uv run python main.py cfg/muscle_tracking_right.py \
  --ckpt pretrained/muscle_tracking/muscle_tracking_right \
  --test
```

### High-Level Muscle-Driven Piano Control

`cfg/muscle.py` trains the piece-specific high-level policy that coordinates
the two low-level muscle controllers. By default, the configuration loads:

```text
pretrained/muscle_tracking/muscle_tracking_left
pretrained/muscle_tracking/muscle_tracking_right
```

Train a high-level muscle-driven policy with:

```bash
uv run python main.py cfg/muscle.py \
  --note notes/017-1_fingering.txt \
  --ckpt <checkpoint_directory>
```

Evaluate the trained policy with the same configuration and note file:

```bash
uv run python main.py cfg/muscle.py \
  --note notes/017-1_fingering.txt \
  --ckpt <checkpoint_directory> \
  --test
```

Use `--device <GPU_ID>` to select a GPU; the default is `0`. Pass
`--device cpu` to run on the CPU. To continue training from the latest
checkpoint in an existing directory, repeat the training command with
`--resume`:

```bash
uv run python main.py cfg/muscle.py \
  --note notes/017-1_fingering.txt \
  --ckpt <checkpoint_directory> \
  --resume
```

Training checkpoints and TensorBoard logs are written to the directory passed
to `--ckpt`.

## Data Attribution

The reference motions are retargeted from the
[ForElise](https://for-elise.github.io/) piano performance dataset. The note
files are extracted from the
[Piano Fingering Dataset (PIG)](https://beam.kisarazu.ac.jp/research/PianoFingeringDataset/).
Please follow the licenses and citation requirements of the source datasets.

## Citation

If you use this code, the hand model, or the provided motion data, please cite:

```bibtex
@article{music,
  author = {Xu, Pei and Ye, Yufei and Sun, Shuchun and Ding, Yu and Schumann, Elizabeth and Liu, C. Karen},
  title = {{MUSIC}: Learning Muscle-Driven Dexterous Hand Control},
  journal = {ACM Transactions on Graphics},
  volume = {45},
  number = {4},
  year = {2026},
  doi = {10.1145/3811402}
}
```

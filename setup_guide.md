# Kidney Stone Segmentation Setup Guide

This project notebook is configured to run on Python 3.12.10 in the `MImage-project` virtual environment.

## 1. Create the virtual environment

From the project root in PowerShell:

```powershell
py -3.12 -m venv MImage-project
```

If `py -3.12` is not available, use an installed Python 3.12 executable instead.

## 2. Activate the environment

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\MImage-project\Scripts\Activate.ps1
```

## 3. Upgrade packaging tools

```powershell
python -m pip install --upgrade pip setuptools wheel
```

## 4. Install required libraries

The notebook imports these third-party packages:
- `torch`
- `torchvision`
- `torchaudio`
- `numpy`
- `matplotlib`
- `Pillow`
- `scikit-learn`
- `tqdm`

Install them with:

```powershell
pip install numpy matplotlib Pillow scikit-learn tqdm
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

If you want the CPU-only PyTorch build instead, use the default PyPI install command rather than the CUDA index URL.

## 5. Register the venv as a Jupyter kernel

```powershell
python -m ipykernel install --user --name MImage-project --display-name "Python (MImage-project)"
```

## 6. Verify the environment

```powershell
python --version
python -c "import torch, numpy, matplotlib, PIL, sklearn, tqdm; print('Environment OK')"
```

## 7. Run the notebook

Start Jupyter and open `code.ipynb`:

```powershell
jupyter notebook
```

or

```powershell
jupyter lab
```

## Notes

- The active workspace venv is Python 3.12.10 at `MImage-project`.
- The notebook’s configuration cell prints GPU information and was fixed to use `torch.cuda.get_device_properties(0).total_memory`.
- The notebook expects the dataset under `data/image` and `data/label`.

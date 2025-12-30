# Unsloth - Google Gemma3-270m fine-tuning notes

## My test config
- Ubuntu 22.04
- RTX 3070 8G
- Pytorch version please refer to official (https://pytorch.org/)
## Python packages
```
unsloth                   2025.12.9
unsloth_zoo               2025.12.7
torch                     2.9.1
torchao                   0.15.0
torchvision               0.24.1
accelerate                1.12.0
bitsandbytes              0.49.0
peft                      0.18.0
trl                       0.24.0
triton                    3.5.1
transformers              4.57.3
xformers                  0.0.33.post2
```

# Need to install the nvidia gfx driver first
```
sudo apt update && sudo apt upgrade -y
ubuntu-drivers devices
sudo ubuntu-drivers autoinstall
reboot
```

# Check the CUDA version
```
nvidia-smi
```
```
$ nvidia-smi
Tue Dec 30 09:58:01 2025       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.95.05              Driver Version: 590.48.01      CUDA Version: 13.1     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 3070        Off |   00000000:01:00.0 Off |                  N/A |
| 37%   34C    P8             21W /  220W |       9MiB /   8192MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A            1891      G   /usr/bin/gnome-shell                      3MiB |
+-----------------------------------------------------------------------------------------+
```

# Check pytorch is workalbe
- torch.cuda.is_available() should return True
```
python3 -c "import torch
print(torch.cuda.is_available())"
True
```

# Install unsloth package
```
pip install accelerate bitsandbytes peft trl triton torchao
pip install unsloth
```

# Start training script
```
python3 chatbot_finetune.py
```

# Running GGUF via llama.cpp, example
```
./llama-cli -m gemma-3-270m-it.Q8_0.gguf
```


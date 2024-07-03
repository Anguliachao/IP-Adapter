conda create -n diffusers python=3.10

conda activate diffusers

#install diffusers from source
#cd /cephFS/yangying/AIGC2024
git clone https://github.com/huggingface/diffusers
#cd /cephFS/yangying/AIGC2024/diffusers
pip install -e .
cd /cephFS/yangying/AIGC2024/diffusers/examples/controlnet/

#install other pkgs
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available())"
pip install xformers
python -c "import xformers;import torch; print(torch.cuda.is_available())"

cd /cephFS/yangying/AIGC2024/ipadapter/IP-Adapter
pip install webdataset
pip install einops
pip install opencv-python==4.8.1.78
python -c "import cv2"

pip install wandb
pip install bitsandbytes

python -c "import wandb;import tensorboard; import accelerate;import torchvision; import triton; import xformers;import transformers"

#installed machine: 202.168.101.210, 202.168.101.119
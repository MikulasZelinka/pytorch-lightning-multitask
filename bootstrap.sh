ENV_NAME=pytorch-lightning-multitask

conda install mamba -n base -c conda-forge
mamba create -n $ENV_NAME python=3.10
conda activate $ENV_NAME

mamba install pytorch torchvision torchaudio cpuonly -c pytorch

mamba env export --from-history > env_initial.yaml

pip install pytorch-lightning sentence-transformers

mamba env export > env_full.yaml
FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel

ARG UID 1000
ENV CUDA cu111
ENV TORCH 1.8.0

# install essential softwares
RUN apt update && apt install -y vim zsh git ssh sudo language-pack-en
RUN update-locale LANG=en_US.UTF-8

# link to python
RUN ln -s /opt/conda/bin/python /usr/bin/python

# python -m pip install packages
RUN python -m pip install --upgrade pip
RUN python -m pip install numpy matplotlib pylint tqdm sentencepiece transformers scikit-learn tensorboard spacy==2.3.7 scispacy black

# install optuna
RUN sudo apt update && sudo apt install -y libssl-dev python3-dev libmysqlclient-dev && python -m pip install optuna mysqlclient scikit-optimize

# install scispaCy
RUN python -m pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.5/en_core_sci_sm-0.2.5.tar.gz

# setup apex
# WORKDIR /tmp/unique_for_apex
# RUN SHA=ToUcHMe git clone https://github.com/NVIDIA/apex.git
# WORKDIR /tmp/unique_for_apex/apex
# RUN pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .
# WORKDIR /workspace

# install pytorch geometric
RUN python -m pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
RUN python -m pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
RUN python -m pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
RUN python -m pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
RUN python -m pip install torch-geometric

RUN adduser --uid $UID --disabled-password user

USER user

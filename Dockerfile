FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# pip install 
RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.1+cu117.html
RUN pip install  dgl -f https://data.dgl.ai/wheels/cu118/repo.html
RUN pip install -r requirements.txt
WORKDIR /workspace/sim_hgcl
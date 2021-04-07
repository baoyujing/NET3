# Network of Tensor Time Series
This is the PyTorch implementation of the paper:

Baoyu Jing, Hanghang Tong and Yada Zhu, [Network of Tensor Time Series](https://arxiv.org/abs/2102.07736), WWW'2021 

## Requirements
- numpy>=1.19.5
- scipy>=1.5.4
- PyYAML>=5.4.1
- tensorly>=0.5.1
- tqdm>=4.59.0
- pandas>=1.1.5
- torch>=1.6.0 
- torchvision>=0.7.0

Packages can be installed via: ```pip install -r requirements.txt```


## Data Preparation
1. *Formulation.*
   Formulate the co-evolving time series (or multi-variate time series) as a tensor time series. 
   The temporal snapshot should be an M-dimensional tensor. 
   Note that vector and matrix are special cases of tensor.
2. *Normalization.* 
   For each single time series within the tensor time series, use z-score of the training split to normalize the values.
3. *Graph construction.* 
   The m-th dimension of the tensor can be associated with a graph, which is represented by the adjacency matrix ![equation](http://www.sciweavers.org/tex2img.php?eq=A_%7Bm%7D%20&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0).
   The adjacency matrix should be normalized by ![equation](http://www.sciweavers.org/tex2img.php?eq=%5Cwidetilde%7BA%7D%20%3D%20%20D%5E%7B-%20%5Cfrac%7B1%7D%7B2%7D%20%7D%20A_mD%5E%7B-%20%5Cfrac%7B1%7D%7B2%7D%20%7D%20&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0).
   Note that if a dimension is not associated with a network, then use the identity matrix.
4. Store the values of the tensor time series and the adjacency matrices in ```values.pkl``` and ```networks.pkl```. 
   Store the indicators for training, validation and testing in ```train_idx.pkl```, ```val_idx.pkl``` and ```test.pkl```.

## Training
1. Specify the mode for training: ```train``` (only training) or```train-eval``` (evaluating the model after each epoch).
2. Specify the task: ```missing``` (missing value recovery) and ```future``` (future value prediction).
3. Specify the paths of the configurations for the model and training.

```python main.py -cm ./configs/model.yml -cr ./configs/run_missing.yml -m train -t missing```

## Evaluation
1. Specify the mode: ```eval```
2. Specify the task: ```missing``` (missing value recovery) and ```future``` (future value prediction).
3. Specify the paths of the configurations for the model and evaluation.

```python main.py -cm ./configs/model.yml -cr ./configs/run_missing.yml -m eval -t missing```

## Citation
Please cite the following paper, if you find the repository or the paper useful.

Baoyu Jing, Hanghang Tong and Yada Zhu, [Network of Tensor Time Series](https://arxiv.org/abs/2102.07736), WWW'2021 

```
@article{jing2021network,
  title={Network of Tensor Time Series},
  author={Jing, Baoyu and Tong, Hanghang and Zhu, Yada},
  journal={arXiv preprint arXiv:2102.07736},
  year={2021}
}
```
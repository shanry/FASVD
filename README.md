# FASVD
Feature Augmentation by SVD

# Dependency
requirements.txt

## Organize data directory as follows
.
├── data  \
│   ├── citeseer  \
│   ├── cora  \
│   ├── flowers  \
│   │   ├── daisy  \
│   │   ├── dandelion  \
│   │   ├── rose  \
│   │   ├── sunflower  \
│   │   └── tulip  \
│   ├── flowers_svd40  \
│   │   ├── daisy  \
│   │   ├── dandelion  \
│   │   ├── rose  \
│   │   ├── sunflower  \
│   │   └── tulip  

## Node classification
python --nc.py --dataset cora --words --c_best 1  \
python --nc.py --dataset cora --vecs --c_best 0.5  \
python --nc.py --dataset cora --svd --c_best 100  \
python --nc.py --dataset cora --esvd --c_best 1  \
python --nc.py --dataset cora --esvd2 --c_best 1  

python --nc.py --dataset citeseer --words  --c_best 0.1  \
python --nc.py --dataset citeseer --vecs  --c_best 0.1  \
python --nc.py --dataset citeseer --svd  --c_best 20  \
python --nc.py --dataset citeseer --esvd  --c_list 10 15  \
python --nc.py --dataset citeseer --esvd2  --c_list 5 10  

## Image classification
run the notebook classify.ipynb


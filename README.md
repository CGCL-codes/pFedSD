# pFedSD

This is the implementation of "Personalized Edge Intelligence via Federated Self- Knowledge Distillation", also published in IEEE TPDS journal.

## Requirements
Please run the following commands below to install dependencies.

```bash
conda create -y -n fed_d python=3.7
conda activate fed_d
conda install -y -c pytorch pytorch=1.7.1 torchvision=0.8.2 matplotlib python-lmdb cudatoolkit=11.3 cudnn
pip install transformers datasets pytreebank opencv-python torchcontrib gpytorch 
```

## Training and evaluation
Details for each argument are included in `./parameters.py`.  

The setup of the FedAvg/FedProx for resnet-8 on cifar10 with pathological distribution:
```bash
python run_gloo.py \
    --arch resnet8 --complex_arch master=resnet8,worker=resnet8 --experiment demo \
    --data cifar10 --pin_memory True --batch_size 64 --num_workers 2 --prepare_data combine \
    --partition_data pathological --shard_per_user 2 \
    --train_data_ratio 0.8 --val_data_ratio 0.0 --test_data_ratio 0.2 \
    --n_clients 10 --participation_ratio 0.6 --n_comm_rounds 5 --local_n_epochs 5 \
    --world_conf 0,0,1,1,100 --on_cuda True \
    --fl_aggregate scheme=federated_average \
    --optimizer sgd --lr 0.01 --local_prox_term 0 --lr_warmup False --lr_warmup_epochs 5 --lr_warmup_epochs_upper_bound 150 \
    --lr_scheduler MultiStepLR --lr_decay 0.1 \
    --weight_decay 1e-5 --use_nesterov False --momentum_factor 0.9 \
    --track_time True --display_tracked_time True --python_path $HOME/anaconda3/envs/fed_d/bin/python \
    --manual_seed 7 --pn_normalize True --same_seed_process True \
    --algo fedavg \
    --personal_test True \
    --port 20001 --timestamp $(date "+%Y%m%d%H%M%S")
```

The setup of the pFedSD for simple cnn on cifar10 with dirichlet distribution:
```bash
python run_gloo.py \
    --arch simple_cnn --complex_arch master=simple_cnn,worker=simple_cnn --experiment demo \
    --data cifar10 --pin_memory True --batch_size 64 --num_workers 2 --prepare_data combine \
    --partition_data non_iid_dirichlet --non_iid_alpha 0.1 \
    --train_data_ratio 0.8 --val_data_ratio 0.0 --test_data_ratio 0.2 \
    --n_clients 10 --participation_ratio 0.6 --n_comm_rounds 5 --local_n_epochs 5 \
    --world_conf 0,0,1,1,100 --on_cuda True \
    --fl_aggregate scheme=federated_average \
    --optimizer sgd --lr 0.01 --local_prox_term 0 --lr_warmup False --lr_warmup_epochs 5 --lr_warmup_epochs_upper_bound 150 \
    --lr_scheduler MultiStepLR --lr_decay 0.1 \
    --weight_decay 1e-5 --use_nesterov False --momentum_factor 0.9 \
    --track_time True --display_tracked_time True --python_path $HOME/anaconda3/envs/fed_d/bin/python \
    --manual_seed 7 --pn_normalize True --same_seed_process True \
    --algo pFedSD \
    --personal_test True \
    --port 20002 --timestamp $(date "+%Y%m%d%H%M%S")
```

## Citation

TODO

## Acknowledgements

The skeleton codebase in this repository was adapted from FedDF[1].

[1] T. Lin, L. Kong, S. U. Stich, and M. Jaggi, “[Ensemble distillation for robust model fusion in federated learning](https://proceedings.neurips.cc/paper/2020/file/18df51b97ccd68128e994804f3eccc87-Paper.pdf),” in NeurIPS, 2020.


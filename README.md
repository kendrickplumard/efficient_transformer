# How To Use

git clone https://github.com/kendrickplumard/efficient_transformer.git

cd efficient_transformer/

pip install -e .

pip install -r requirements.txt

-> On colab you can install dependencies manually if you face any issue by doing: 
- pip install schedulefree (for more infos about the package please consider [this](https://github.com/facebookresearch/schedule_free))
- pip install wandb (more infos [here](https://docs.wandb.ai/quickstart))
- pip install --pre torchao-nightly --index-url https://download.pytorch.org/whl/nightly/cu121 (more infos [here](https://github.com/pytorch/ao/tree/main/torchao/prototype/low_bit_optim))


cd efficient_transformer_v1

python data/shakespear_char/shakespear_char.py (preprocess the dataset)

python train.py (launch the default model with the default config set in config/train_config.py)


Note: For now i decided to make each folder (architecture proposal) self sufficient. This let me all the freedom to explore different things along the way. 






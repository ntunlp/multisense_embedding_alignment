# SaBERT
This the code for the SaBERT model described in our paper.

## install depedencies
```
pip install -r requirements.txt
```


## commands
Please modify PATH_TO in the below script accordingly.

### convert data to TFRecord format

```
bash bin/data.sh
```

### pretrain model
Hyperparameters are saved in bin/bert_config.json.

```
bash bin/pretrain.sh
```

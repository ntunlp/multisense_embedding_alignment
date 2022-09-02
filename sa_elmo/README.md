# SaELMo
This the code for the SaELMo model described in our paper.


## install dependecies
Install python version 3.5 or later, then run the below command to install other dependencies.

```
pip install -r requirements.txt 
```

## commands
Please modify PATH_TO in the below script accordingly.

### pretrain SaELMo monolingual model
Below is an example to pretrain English monolingual model.

```
bash examples/run_en.sh 
```

Dump weight.

```
bash examples/dumpweight.sh
```

### pretrain SaELMo bilingual model
Below is an example to pretrain English-German bilingual model.

```
bash example/run_bilingual_de.sh
```

Dump weight.
```
bash examples/dumpweight_bi.sh
```

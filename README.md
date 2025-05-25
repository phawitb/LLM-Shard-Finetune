# LLM-Shard-Finetune

### 1.Download Dataset (For first time)
```
python load_wikitext_to_csv.py
```

### 2.Download Model (For first time)
```
python load_model.py
```
### 3.Run Edges (Workers)
```
python shard_worker.py 0
python shard_worker.py 1
python shard_worker.py 2
python shard_worker.py 3
python shard_worker.py 4
```

### 4.Run Server (For control Finetune)
```
python server.py 
```

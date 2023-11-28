# Yandex_cup (placw: 20/236)
Hackathon from Yandex_cup, ML direction: RecSys "In search of reality", in the task "Classification with multiple levels of music genres". It was necessary to train the best model to predict the top of several classes (out of 256 possible), genres of songs thanks to embeddings.

- https://contest.yandex.ru/contest/54251/problems/
## Data:
Data Form: (100000, N, 768), where *N* is different size for each object.
- track_embeddings.tar.gz — contains files with track embeddings (type = np.float32). Each track is described by a sequence of embeddings of dimension 768, calculated from a fragment of an audio track of fixed length. 
- train.csv — contains track id and a comma-separated list of tags, 256 tags in total;
- test.tsv — the same format but without a list of tags;
- Baseline.ipynb — an example of a laptop with a naive model;
- sample_submission.csv — sample solution.
## New idea: 
- Identification and addition of new features (min, max, variance, standard deviation, correlation/difference by lines)
- Training of the best models (FC, CNN, LSTM)
- Using K-fold validation during training
- Combining all models into blending, which gave the best result (stacking retraining)
## Technologies:
- Pytorch
- Transformers
- RNN
- KFold
- Seaborn
## Specificity
- Unusual data distribution (each object had a different size shape=(N, 768))
- Using a large calorite of proven models
- K-fold training with blending

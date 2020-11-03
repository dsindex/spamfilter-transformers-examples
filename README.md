### spam filter examples using transformers and trainer

#### 실험 환경

- [conda 설치](https://docs.anaconda.com/anaconda/install/mac-os/#using-the-command-line-install)

- 기본 설정
$ conda create -n study python=3.6
$ conda activate study

- 필요한 패키지 설치 
$ python -m pip install -r requirements

#### 실험

- 학습 및 평가
$ python train.py

- 결과
  - distilbert-base-cased, F1-SCORE(macro) : 90.95%
```
              precision    recall  f1-score   support

         ham     0.9845    0.9695    0.9769       131
        spam     0.8000    0.8889    0.8421        18

    accuracy                         0.9597       149
   macro avg     0.8922    0.9292    0.9095       149
weighted avg     0.9622    0.9597    0.9606       149

[[127   4]
 [  2  16]]

```
  - bert-base-cased, F1-SCORE(macro) : 93.68%
```
              precision    recall  f1-score   support

         ham     0.9847    0.9847    0.9847       131
        spam     0.8889    0.8889    0.8889        18

    accuracy                         0.9732       149
   macro avg     0.9368    0.9368    0.9368       149
weighted avg     0.9732    0.9732    0.9732       149

[[129   2]
 [  2  16]]
```

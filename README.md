## env

- conda
  - [conda](https://docs.anaconda.com/anaconda/install/mac-os/#using-the-command-line-install)
  ```
  $ conda create -n study python=3.6
  $ conda activate study
  ```

- requirements
```
$ python -m pip install -r requirements
```

## sequence classification

- data samples
  - http://www.dt.fee.unicamp.br/~tiago/smsspamcollection/smsspamcollection.zip

- train/evaluate
```
$ python sequence-classification.py
```

- hyper-parameter search
```
$ python sequence-classification.py --hp_search_ray --eval_steps=500 --hp_dashboard_port={your_port}
...
Number of trials: 10 (1 PAUSED, 6 PENDING, 1 RUNNING, 2 TERMINATED)
+------------------------+------------+------------------+-----------------+--------------------+-------------------------------+----------+----------------+-------------+
| Trial name             | status     | loc              |   learning_rate |   num_train_epochs |   per_device_train_batch_size |     seed |   weight_decay |   objective |
|------------------------+------------+------------------+-----------------+--------------------+-------------------------------+----------+----------------+-------------|
| _objective_e60e7_00000 | TERMINATED |                  |     5.61152e-06 |                  2 |                             8 | 38.0779  |      0.183556  |    0.886667 |
| _objective_e60e7_00001 | TERMINATED |                  |     2.91064e-05 |                  4 |                            32 | 24.3477  |      0.0418482 |    0.98     |
| _objective_e60e7_00002 | PAUSED     |                  |     2.05134e-06 |                  5 |                             4 |  7.08379 |      0.0876434 |    0.973333 |
| _objective_e60e7_00003 | RUNNING    | 9.1.40.130:37137 |     1.30667e-06 |                  3 |                             4 | 34.7809  |      0.109909  |    0.933333 |
| _objective_e60e7_00004 | PENDING    |                  |     1.59305e-05 |                  2 |                            16 | 28.6148  |      0.136821  |             |
| _objective_e60e7_00005 | PENDING    |                  |     1.09943e-06 |                  1 |                             4 | 38.8265  |      0.235553  |             |
| _objective_e60e7_00006 | PENDING    |                  |     4.62259e-05 |                  4 |                            16 |  9.28123 |      0.0599021 |             |
| _objective_e60e7_00007 | PENDING    |                  |     2.3102e-06  |                  2 |                            16 |  8.15278 |      0.15427   |             |
| _objective_e60e7_00008 | PENDING    |                  |     4.05961e-06 |                  5 |                             4 | 21.4655  |      0.177724  |             |
| _objective_e60e7_00009 | PENDING    |                  |     7.30954e-06 |                  5 |                            32 | 12.3579  |      0.0139351 |             |
+------------------------+------------+------------------+-----------------+--------------------+-------------------------------+----------+----------------+-------------+
...

```

## question answering

- train/evaluate
```
* for metirc for squad_v2, we need to clone transformers
$ cd ..
$ git clone -b v4.1.1 https://github.com/huggingface/transformers.git

$ python question-answering.py --squad_v2 --num_train_epochs=3
...
OrderedDict([('exact', 64.35610208035037), ('f1', 67.47884060584892), ('total', 11873), ('HasAns_exact', 60.07085020242915), ('HasAns_f1', 66.32528247524326), ('HasAns_total', 5928), ('NoAns_exact', 68.6291000841043), ('NoAns_f1', 68.6291000841043), ('NoAns_total', 5945), ('best_exact', 64.35610208035037), ('best_exact_thresh', 0.0), ('best_f1', 67.47884060584907), ('best_f1_thresh', 0.0)])

```


## reference

- https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/text_classification.ipynb?fbclid=IwAR1CiTt_tKSvh4ee_Kpep41yS8Dhd6m9osJYZaRaR5qFuycOvADeCK6jIZA#scrollTo=zVvslsfMIrIh
- https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/question_answering.ipynb#scrollTo=whPRbBNbIrIl
- ray
  - https://docs.ray.io/en/master/tune/examples/pbt_transformers.html
  - https://docs.ray.io/en/latest/tune/api_docs/execution.html#tune-run
  - https://docs.ray.io/en/master/package-ref.html
- https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
```
Micro average (averaging the total true positives, false negatives and false positives) is only shown for multi-label or multi-class with a subset of classes, because it corresponds to accuracy otherwise.
```

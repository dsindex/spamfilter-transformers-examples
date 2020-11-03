### spamfilter examples using transformers and trainer

#### env

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

#### experiments

- train/evaluate
```
$ python train.py
```

- hyper-parameter search
```
$ python train.py --hp_search_ray --hp_server_port=9599
```

- reference
  - https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/text_classification.ipynb?fbclid=IwAR1CiTt_tKSvh4ee_Kpep41yS8Dhd6m9osJYZaRaR5qFuycOvADeCK6jIZA#scrollTo=zVvslsfMIrIh
  - https://docs.ray.io/en/master/tune/examples/pbt_transformers.html
  - https://docs.ray.io/en/latest/tune/api_docs/execution.html#tune-run
  - [classification_report](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html)
  ```
  Micro average (averaging the total true positives, false negatives and false positives) is only shown for multi-label or multi-class with a subset of classes, because it corresponds to accuracy otherwise.
  ```

# Fraud Detection
Evan Frangipane

- [Dataset](#dataset)

## Dataset

We have a kaggle dataset on [credit card
fraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). This is
a highly imbalaced dataset with fraud only $0.172\%$ of $284,807$
transactions.

This dataset is anonymized for personal privacy, so we have PCA
components as our features.

We calculate mutual information and correlation on our features to find
out what is interesting and predictive.

![Mutual Information](images/mutual_info_before.jpg)

![Correlation](images/corr_before.jpg)

# Baseline for SLRTP Track 3: Spotting on BSL Corpus

This repository is intended to produce a baseline method for spotting on BSL Corpus. Using the provided logits in the data for this challenge, the model computes the top-5 predictions at each frame. Given a query gloss, we consider a sign gloss for this query to be localised if amongst the top-5 predictions.  

The structure of this repository also contains a dataloader to load model features and labels, and a simple MLP model that can be trained on the features. 

## To evaluate baseline model using the provided logits on test set: 

```bash test.sh```

This create a file called `res/submission_dev.zip`, which may be uploaded to the CodaLab server. 

## To train a simple MLP model 

```bash train.sh``` 





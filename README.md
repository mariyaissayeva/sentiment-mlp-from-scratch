# Sentiment MLP from Scratch

A 2-layer neural net built in pure NumPy on a review dataset I collected myself.

## What the notebook does

Two things in one project. First, the data: a set of 92 customer reviews of Starbucks based in Astana, collected and annotated by hand (53 negative, 39 positive) from 2GIS website, with two engineered features per review derived from a sentiment lexicon, counting positive and negative cue words.

Second, the model: a small MLP with a 2-2-1 architecture implemented entirely in NumPy, with forward pass, backpropagation, and gradient descent written out by hand rather than taken from a framework. The features are normalized, the data is split 80/20, and the network is trained with binary cross-entropy for up to 1,000 epochs. One cell walks through a single backpropagation step on one sample so every gradient can be inspected by eye. The notebook then compares ReLU, tanh, and sigmoid as hidden activations, training a model with each and plotting their loss curves side by side, and evaluates the classifier with accuracy, precision, recall, an F1 score, and a confusion matrix.

## Results

All three activations converged to the same test accuracy of 63.2%, which is exactly the share of negative reviews in the test set: the network learned to predict the majority class and nothing more, so precision and recall for the positive class are zero. The reason is visible in the data rather than the math: most reviews contain none of the lexicon words, so the two features carry almost no signal. The backpropagation works; the features were the bottleneck. That makes the project an honest demonstration of why feature engineering matters as much as the network behind it.

## Tech

Python, NumPy, pandas

## Notes

All three activations landed on the same 63.2% because most reviews contain none of the lexicon words, so the features carry almost no signal and the network defaults to the majority class. The biggest limitation is the tiny dataset of 92 reviews with a test set of just 19. A larger review corpus and richer text features such as TF-IDF or word embeddings would give the network something to learn. The from-scratch implementation fits best as a teaching tool for backpropagation.

## Personal note

I designed it to demonstrate the knoledge of the process and it was also fun considering I love Starbucks and people were being sooo funny and sincere in their reviews ;))))

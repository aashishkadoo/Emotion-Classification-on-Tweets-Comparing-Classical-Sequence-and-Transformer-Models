# Emotion-Classification-on-Tweets-Comparing-Classical-Sequence-and-Transformer-Models
This project compares three approaches to four-class emotion classification (anger, joy, optimism, sadness) on the TweetEval Emotion benchmark. The aim was to measure what extra performance increasingly complex models actually buy on short, noisy, imbalanced social-media text, and whether the cost is justified.

Approach

Data and evaluation setup: I pooled the official train and validation splits and re-split them once (stratified 90/10, seed 42). All three models used that split, and the official 1,421-tweet test set was left untouched until the final evaluation. I checked for exact and near-duplicate leakage across splits. Because the classes are imbalanced (optimism is under 9% of the data), I used macro-F1 as the headline metric, with accuracy and weighted-F1 alongside it.
Preprocessing: I used a deliberately conservative pipeline. It decoded HTML entities, removed the anonymised @user tokens and kept hashtag words. It did not remove stop words or apply stemming, which preserves negation and keeps distinct word forms separate. I justified this with a stemming example on the sadness features and a comparison of how each model handles out-of-vocabulary words.

Model 1, TF-IDF + Logistic Regression: a class-weighted linear baseline using unigrams and bigrams (9,125 features). I analysed its learned weights to see what vocabulary it relied on.

Model 2, Embeddings + BiLSTM: word embeddings learned from scratch, followed by a bidirectional LSTM, with class weighting and early stopping. I also trained a unidirectional LSTM to test whether bidirectionality helped.

Model 3, Twitter-RoBERTa: cardiffnlp/twitter-roberta-base fine-tuned with class-weighted cross-entropy. I chose a maximum sequence length of 64 from the token-length analysis, since a length of 32 would have truncated 27% of tweets.

Results (official test set)

Model	Accuracy	Macro-F1	Optimism F1	Training time
TF-IDF + Logistic Regression	0.647	0.606	0.420	1.5 s
Embeddings + BiLSTM	0.606	0.579	0.370	20 s
Twitter-RoBERTa	0.797	0.764	0.603	~2 min (GPU)

Key Findings

Twitter-RoBERTa beat the other two models by about 16–18 macro-F1 points, and it improved every class. The largest gain was on the minority optimism class, where F1 rose from about 0.37–0.42 to 0.60.

The BiLSTM, with roughly 32× the parameters of the linear model, scored below TF-IDF on every metric and every class. It overfit quickly, reaching about 97% training accuracy while validation accuracy stalled. With only about 3,300 training tweets, learning embeddings from scratch cost accuracy rather than adding it.
Bidirectionality gave no measurable benefit over a unidirectional LSTM on tweets averaging about 16 words.

Error analysis of the 148 test tweets (10.4%) that all three models got wrong showed recurring failure patterns: sarcasm, quote-like text being read as optimism, mixed emotional signals, ambiguous labels, and cases needing world knowledge. The transformer made its five highest-confidence errors at 98%+ confidence, so high confidence did not mean it was right on these inputs. The simpler TF-IDF model was still right on 114 tweets where the transformer was wrong.

The transformer's roughly 3,400× parameter cost over the baseline was easy to justify here, given the short GPU training time. TF-IDF remains the better choice if interpretability is required or no GPU is available.
Tools

Google Colab,Python, scikit-learn, TensorFlow/Keras, PyTorch, Hugging Face Transformers and Datasets, pandas, NumPy, matplotlib.

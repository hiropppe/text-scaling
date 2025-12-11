import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import pandas as pd
    import scipy as sc
    return (pd,)


@app.cell
def _(pd):
    df = pd.read_csv("./data/YS.2012.csv")
    return (df,)


@app.cell
def _(df):
    df
    return


@app.cell
def _(df, mo):
    # 他との一貫性のために前処理に python を使いたいので BoW からトークン列を復元
    r_vocab = df.columns[4:]
    r_bow = df.iloc[:, 4:]
    docs = []
    with mo.status.progress_bar(total=len(df)) as bar:
        for row in r_bow.iterrows():
            doc = " ".join([" ".join(int(row[1].iloc[i])*[v]) for i, v in enumerate(r_vocab)])
            docs.append(doc)
            bar.update()
    return (docs,)


@app.cell
def _(docs):
    docs[:10]
    return


@app.cell
def _(docs, stats_YS):
    N, M, V, D, W, p_v, vocab, word2id = stats_YS(docs)
    return D, M, N, V, W, p_v, vocab, word2id


@app.cell
def _(vocab):
    vocab
    return


@app.cell
def _(D, M, N, V, W, p_v, vocab, word2id):
    N, M, V, D, W, p_v, vocab, word2id
    return


@app.cell
def _(np):
    from sklearn.feature_extraction.text import CountVectorizer

    def stats_YS(docs):
  
      lowercase=True
      max_df=1.0
      min_df=3
      max_features=None
  
      cv = CountVectorizer(binary=False, lowercase=lowercase, max_df=max_df, min_df=min_df, max_features=max_features)
      BoW = cv.fit_transform(docs).toarray()
  
      word2id = cv.vocabulary_
      vocab = cv.get_feature_names_out()
  
      expand_freq = False
  
      M = len(BoW)
      V = len(vocab)
      if expand_freq:
        # expand n_dv as duplicate rows
        doc_ids, token_ids = [], []
        for d, doc in enumerate(docs):
            for token in doc.strip().split():
                if token in word2id:
                  doc_ids.append(d)
                  token_ids.append(word2id[token])
  
        N = len(token_ids)
        W = token_ids
        D = doc_ids
      else:
        data_index = np.where(BoW.ravel() > 0)[0]
        N = len(data_index)
        N_dv = BoW.ravel()[data_index]
        D = (data_index/V).astype(int)
        W = data_index%V
  
      p_v = np.sum(BoW, axis=0)/np.sum(BoW)
  
      return N, M, V, D, W, p_v, vocab, word2id
    return (stats_YS,)


@app.cell
def _():
    return


@app.cell
def _():
    import numpy as np
    from scipy import optimize
    from scipy import stats
    from typing import Callable

    class MAPEstimator:
        """汎用MAP推定クラス"""
    
        def __init__(self, log_likelihood_fn: Callable, log_prior_fn: Callable):
            self.log_likelihood_fn = log_likelihood_fn
            self.log_prior_fn = log_prior_fn
            self.result_ = None
    
        def fit(self, x0, bounds=None, method='L-BFGS-B'):
            def neg_log_posterior(theta):
                ll = self.log_likelihood_fn(theta)
                lp = self.log_prior_fn(theta)
                return -(ll + lp)
        
            self.result_ = optimize.minimize(
                neg_log_posterior, x0=x0, method=method, bounds=bounds
            )
            return self
    
        @property
        def theta_map(self):
            return self.result_.x if self.result_ else None
    return (np,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

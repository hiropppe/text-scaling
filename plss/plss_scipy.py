import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import scipy as sc
    import gensim.downloader as api

    from gensim.models import KeyedVectors
    from scipy.stats import norm, uniform, pearsonr
    from scipy.optimize import minimize
    from sklearn.feature_extraction.text import CountVectorizer
    return CountVectorizer, KeyedVectors, minimize, np, pd, pearsonr


@app.cell
def _():
    # このノートで使用する主な変数
    ## df: YoungSoroka.2012 を前処理した 文書メタ情報 + BoW DataFrame
    ## corpus: コーパス（トークン列）
    ## D: 文書数
    ## V: 語彙数
    ## N: トークン数
    ## word2id: 語彙ID辞書
    ## vocab: 語彙
    ## wv_model: 単語埋め込み Word2Vec モデル (Gensim KeyedVectors)
    ## dim: 単語埋め込み次元数
    ## BoW = 単語vの文書d内の出現頻度 (DxV次元)、=n_dv
    ## p_v: 単語vの確率（V次元）
    ## beta: 単語埋め込み空間の意味方向ベクトル（dim次元）
    ## v: 単語vの埋め込みベクトル(dim次元)
    ## phi: 単語vの極性（V次元）、PLSS では beta^T*v として計算
    ## theta: 文書dの極性（D次元）、今回の推定対象
    return


@app.cell
def _(pd):
    # YoungSoroka.2012 を前処理した BoW データをロード
    df = pd.read_csv("./data/YS.2012.csv")
    df
    return (df,)


@app.cell
def _(df, mo):
    # BoW を CountVectorizer で調整したいので BoW DataFrame から生トークン列を復元
    def restore_tokens(df):
        vocab = df.columns[4:]
        bow = df.iloc[:, 4:]
        docs = []
        with mo.status.progress_bar(total=len(df)) as bar:
            for row in bow.iterrows():
                doc = " ".join([" ".join(int(row[1].iloc[i])*[v]) for i, v in enumerate(vocab)])
                docs.append(doc)
                bar.update()
        return docs

    # トークン列を復元
    corpus = restore_tokens(df)
    corpus[:10]
    return (corpus,)


@app.cell
def _(CountVectorizer, corpus, np):
    def analyze_corpus(corpus):
        """ コーパスを解析
            Returns:
                BoW: 文書x単語行列（出現頻度）
                word2id: 語彙ID辞書
                vocab: 語彙
                p_v: 単語の出現確率
        """

        lowercase=True
        max_df=1.0
        min_df=3
        max_features=None

        cv = CountVectorizer(binary=False, lowercase=lowercase, max_df=max_df, min_df=min_df, max_features=max_features)
        BoW = cv.fit_transform(corpus).toarray()

        word2id = cv.vocabulary_
        vocab = cv.get_feature_names_out()

        D: int = len(corpus)
        V: int = len(vocab)
        N: int = BoW.sum()
        p_v = np.sum(BoW, axis=0)/np.sum(BoW)

        return BoW, word2id, vocab, p_v

    BoW, word2id, vocab, p_v = analyze_corpus(corpus)
    return BoW, p_v, vocab


@app.cell
def _(KeyedVectors):
    #wv_model = api.load("glove-wiki-gigaword-300")
    #wv_model.save("./data/glove-wiki-gigaword-300.kv")
    wv_model = KeyedVectors.load("./data/glove-wiki-gigaword-300.kv")
    return (wv_model,)


@app.cell
def _(np, vocab, wv_model):
    def get_plss_phi(pos_words, neg_words, wv, vocab):
        """ 両極のシード単語群から得られる２つの平均ベクトルから意味方向ベクトルを計算
        """

        normed_vec = wv.get_normed_vectors()
        key2index = wv.key_to_index

        def calc_beta(pos_words, neg_words):
            def mean_vec(words):
                return np.array([normed_vec[key2index[v]] for v in words if v in key2index]).mean(axis=0)

            beta = mean_vec(pos_words) - mean_vec(neg_words)
            beta = beta/np.linalg.norm(beta)
            return beta

        beta = calc_beta(pos_words, neg_words)  
        phi = np.array([beta @ normed_vec[key2index[v]] if v in key2index else 0 for v in vocab])
        return phi

    pos_words = ['good', 'nice', 'excellent', 'positive', 'fortunate', 'correct', 'superior']
    neg_words = ['bad', 'nasty', 'poor', 'negative', 'unfortunate', 'wrong', 'inferior']

    # 単語極性の計算
    phi = get_plss_phi(pos_words, neg_words, wv_model, vocab)
    return (phi,)


@app.cell
def _(BoW, minimize, np, p_v, phi, vocab):
    def logsumexp(x):
        """数値安定な log-sum-exp の計算
        log(∑ exp(x_i)) = max(x) + log(∑ exp(x_i - max(x)))
        """
        x_max = np.max(x)
        return x_max + np.log(np.sum(np.exp(x - x_max)))

    def estimate_theta(BoW, p_v, phi):
        """文書極性θの推定"""
        D = len(BoW)
        V = len(vocab)
        theta = np.zeros(D)

        # log p(v) を事前に計算
        log_p_v = np.log(p_v + 1e-10)

        for d in range(D):
            # 文書 d の対数尤度（負の対数事後確率）
            def neg_log_posterior(theta_d):
                # log p(v|θ, φ) の計算（式9）
                # logits = log p(v) + θ * φ_v
                logits = log_p_v + theta_d * phi

                # 数値安定な log-sum-exp
                log_normalizer = logsumexp(logits)

                # 正規化された対数確率
                log_probs = logits - log_normalizer

                # 尤度： ∑_v n_dv * log p(v|θ_d, φ)
                llik = np.sum(BoW[d, :] * log_probs)

                # 事前分布の対数： θ ~ N(0, 1)
                # log p(θ) = -0.5 * log (2π) - 0.5 * θ^2
                #          = -0.5 * θ^2 + const
                prior = -0.5 * theta_d**2

                # 負の対数事後確率（最小化するため符号反転）
                return -(llik + prior)

            # 最適化
            res = minimize(
                neg_log_posterior,
                x0=0.0, # 初期値
                method="L-BFGS-B"
            )

            theta[d] = res.x[0]
        return theta

    theta_estimated = estimate_theta(BoW, p_v, phi)
    theta_estimated
    return (theta_estimated,)


@app.cell
def _(df, pearsonr, theta_estimated):
    correlation, p_value = pearsonr(theta_estimated, df["ys_scale"].to_list())
    print(f"相関係数: {correlation}, p値: {p_value}")
    return


@app.cell
def _(df, np, theta_estimated):
    import matplotlib.pyplot as plt

    # 人手アノテーションのスケールを取得
    ys_scale = df["ys_scale"].values

    # 1. 散布図：推定θ vs 人手スケール
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # 散布図
    axes[0].scatter(ys_scale, theta_estimated, alpha=0.5)
    axes[0].set_xlabel("Human Annotation (ys_scale)")
    axes[0].set_ylabel("Estimated θ")
    axes[0].set_title("Estimated θ vs Human Scale")

    # 相関係数を表示
    corr = np.corrcoef(ys_scale, theta_estimated)[0, 1]
    axes[0].text(0.05, 0.95, f"r = {corr:.3f}", transform=axes[0].transAxes,
               fontsize=12, verticalalignment='top')

    # 2. Boxplot：スケール別のθ分布
    scale_groups = [theta_estimated[ys_scale == s] for s in [1, 2, 3, 4, 5]]
    axes[1].boxplot(scale_groups, labels=["1\n(Neg)", "2", "3\n(Neu)", "4", "5\n(Pos)"])
    axes[1].set_xlabel("Human Scale")
    axes[1].set_ylabel("Estimated θ")
    axes[1].set_title("θ Distribution by Human Scale")

    # 3. ヒストグラム：θの分布
    axes[2].hist(theta_estimated, bins=30, edgecolor='black', alpha=0.7)
    axes[2].set_xlabel("Estimated θ")
    axes[2].set_ylabel("Frequency")
    axes[2].set_title("Distribution of Estimated θ")
    axes[2].axvline(x=0, color='red', linestyle='--', label='θ=0')
    axes[2].legend()

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 補講
    標準正規分布の確率密度関数

    $$p(\theta) = \frac{1}{\sqrt{2\pi}} \exp\left(-\frac{\theta^2}{2}\right)$$

    対数を取る

    $$\log p(\theta) = \log\left(\frac{1}{\sqrt{2\pi}}\right) + \log\left(\exp\left(-\frac{\theta^2}{2}\right)\right)$$

    $$= -\frac{1}{2}\log(2\pi) - \frac{\theta^2}{2}$$

    $$= -\frac{1}{2}\theta^2 + \text{const}$$
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

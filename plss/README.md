# 確率的潜在意味スケーリング (PLSS)
LSS を項目反応理論 (IRT) をもとに統計モデルとして確率化したテキストの潜在的な尺度に対応する連続値を扱うことのできるモデル

確率的潜在意味スケーリング (Probabilistic Latent Semantic Scaling)  
http://chasen.org/~daiti-m/paper/nl249plss.pdf

## 項目反応理論 (IRT)
心理統計学の分野で開発された方法論で、テストの結果から被験者の潜在的な能力あるいは傾向 $\theta$ を推定することを目的とする。被験者の $\theta$ は標準正規分布 $\mathcal{N}(0,1)$ に従っていると仮定する。

\begin{align}
\theta \sim \mathcal{N}(0, 1) \tag{4}
\end{align}

被験者 $i$ の問題 $j$ 対する正答確率 $p_{ij}$ は次のようにモデル化される。

\begin{align}
p_{ij} = \sigma(\alpha_j(\theta_i-\beta_j)) \tag{5}
\end{align}

$\sigma(x)=\frac{1}{1+e^{-x}}$ は、ロジスティック（シグモイド）関数。問 $j$ への正答率は問題の難易度を表す $\beta_j$ を境に、$\alpha_j$ に比例する確率で上昇することを表している。

観測データとして$Y = \{y_{ij}\}_{i,j}$ を考え、$y_{ij}$ は被験者$i$が問 $j$ に正答した時 1, 誤ったときに 0 とすると、$Y$ の確率は下記のようにあらわすことができる。

\begin{align}
p(Y|\alpha, \beta, \theta) &= \prod_i\prod_j Bernoulli (y_{ij}|p_{iij}) \tag{6} \\
&= \prod_i\prod_j Bernoulli (y_{ij}|\sigma(\alpha_j(\theta_i-\beta_j))) \tag{7}
\end{align}


IRT では事前確率 (4) の下で尤度 (7) を最大にするパラメータ ${\alpha_i, \beta_j}$ を求め、同時に被験者の能力 ${\theta_i}$ を推定する。

## PLSS の生成過程
PLSS ではまず式 (5) を多項分布に拡張する形で、テキストにおける単語 $v$ の確率を次のようにモデル化する。

\begin{align}
p(v|\theta,\phi) &\propto p(v)exp(\theta \cdot \phi_v) \tag{8} \\
&= \frac{exp(\log p(v) + \theta \cdot \phi_v)}{\sum^V_{v=1}exp(\log p(v)+\theta \cdot \phi_v)} \tag{9}
\end{align}

Wordfish 同様に $\phi_v \in \mathbb{R}$ は、単語 $v$ の潜在的な極性、$\theta \sim \mathcal{N}(0, 1)$ は、テキストの潜在的な極性を表す。  
このモデルは、IRT の多項分布化、あるいは式 (9) で表される多項ロジスティック回帰において、$\theta$ も $\phi$ も未知の場合の教師なし学習とみなすことができる。

このとき、テキスト $d$ の確率は単語 $v$ のテキスト内での頻度を $n_{dv}$ とおくと

\begin{align}
p(d|\theta,\phi) = \prod^V_{v=1} p(v|\theta,\phi)^{n_{dv}} \tag{10}
\end{align}

と書けるので、$D$ 個のテキストからなるコーパス $D$ 全体の確率は、

\begin{align}
p(D|\Theta,\phi) = \prod^{D}_{d=1}\prod^{V}_{v=1} p(v|\theta_d,\phi)^{n_{dv}} \tag{11}
\end{align}

と表せる。

式 (11) は、mcmc や最適化でパラメータを推定することが可能だが、$\phi_1,\cdots,\phi_V$ を独立に学習するため、単語間の意味的関係を扱えないという問題がある。
PLSS では、単語極性 $\phi$ をニューラル単語ベクトルを用いて設計する事でこれに対処している。

## 単語極性の設計
PLSS では、を $K$ 次元のニューラル単語ベクトル $\vec{v}$ を用いて、$\phi_v$ を下記の通り計算する。

\begin{align}
\phi_v = \boldsymbol{\beta}^T\vec{v}
\end{align}

これにより、各単語の $\phi$ を独立に推定する代わりに $K$ 次元の係数ベクトル $\boldsymbol{\beta}$ を１つ推定するだけでよくなる。 
この $\boldsymbol{\beta}$ は、単語埋め込みベクトル空間における"良い-悪い", "右翼-左翼"のような $\theta$ の極性を与える「意味方向（極性軸）」を表している。  

これによって、式 (9) は、

\begin{align}
p(v|\theta,\boldsymbol{\beta}) &= \frac{exp(\log p(v) + \theta \cdot \boldsymbol{\beta}^T\vec{v})}{\sum^V_{v=1}exp(\log p(v)+\theta \cdot \boldsymbol{\beta}^T\vec{v})} \tag{12}
\end{align}

と表すことができる。  

$\theta$ と $\boldsymbol{\beta}$ を式 (11) に基づいて、教師なしで同時推定することも可能だが、これによって学習される $\boldsymbol{\beta}$ は、Wordfish 同様に分析の目的に一致しているとは限らないとして、より直接的に単語極性の設計している。  

論文では、分析目的の対応した正と負の極性単語集合が与えられた時に、意味方向 $\boldsymbol{\beta}$  を計算する方法として、以下の二つをあげている。

- 超密埋め込み (Ularadense embedding) [DensRay](https://arxiv.org/abs/1904.08654) 
- 単純に負の極性語の平均ベクトルから正の極性語の平均ベクトルへの方向ベクトルを用いる方法

### DensRay による意味方向の設計
＃ ここわかってない

語彙集合 $V := \{v_1, v_2, ..., v_n\}$ とその埋め込み $E \in \mathbb{R}^{n \times d}$ および、言語的特徴アノテーション $l:V \to \{-1,1\}$ が与えられた時に、$EQ$ が解釈可能となるような、直交行列 $Q \in \mathbb{R}^{d \times d}$ を求めることを目的とする。   
これによって求まった、$EQ$ の最初の $k$ 次元を解釈可能な超高密度の単語空間 (interpretable ultradense word space) と呼ぶ。  

DensRay では、言語的信号 $L_=:= \{(v, w) \in V \times V |\; l(v) = l(w)\}$ および $L_{\not =}:= \{(v, w) \in V \times V |\; l(v) {\not =} l(w)\}$ が与えられた上で、[Densifier](https://arxiv.org/abs/1602.07572) を修正した、以下の目的関数について最適化する。

\begin{align}
\max_q \sum _ {(v,w) \in L_{\not =}} \alpha_{\not =} &\left\| q^\intercal d_{vw} \right\| _ { 2 }^2- \nonumber \\
&\sum _ { (v,w) \in L_{=}}  \alpha_{=}  \left\| q^\intercal d_{vw} \right\| _ { 2 }^2 \tag{13}
\end{align}

$d_{vw}:= e_v - e_w$, $e_i$ は、$E$ の $i$番目の行ベクトル。$\text{subject to } q^\intercal q = 1$, $q \in \mathbb{R}^d$. $\alpha_{\not =}, \alpha_{=}  \in [0, 1]$ はハイパーパラメータ。

$\left\|x\right\|_2^2 = x^\intercal x$ と行列積の結合法則を用いると式 (13) は次の通り単純化できる。

\begin{align}
\max_{q}\; & q^\intercal \Big( \alpha_{\not =}  \sum_{(v,w) \in L_{\not =}} d_{vw}d_{vw}^\intercal - \\
&\qquad\qquad \alpha_{=}  \sum_{(v,w) \in L_{=}} d_{vw}d_{vw}^\intercal \Big) q \nonumber \tag{14} \\
& =:\; \max_{q}\; q^\intercal  A q  \quad \text{subject to } q^\intercal q = 1. \nonumber 
\end{align}


これは、$A$ と $q$ のレイリー商を最大化することを指す。$A$ は実対称行列である。このとき $A$ の固有ベクトルを、対応する固有値の大きい順に並べた行列が目的の行列 $Q$ となる。$A$ が実対称行列なので $Q$ は常に直交する。

$k = 1$ としたとき、$Q$ と $EQ$ のそれぞれ 1 次元目が $\boldsymbol{\beta}$ と $\phi$ に対応する。

LSS に付属する以下の標準的な極性辞書を用いて計算された単語の極性値を参考レベルで記載する。  

\begin{align}
S_+ &:= \{'good', 'nice', 'excellent', 'positive', 'fortunate', 'correct', 'superior'\} \\
S_\_ &:= \{'bad', 'nasty', 'poor', 'negative', 'unfortunate', 'wrong', 'inferior'\}
\end{align}

### 平均ベクトル間の方向ベクトルによる意味方向の設計
PLSSでは、$S_+$ を正の極性語集合、$S_\_$ は負の極性語集合を使って、次のように単純な計算式で $\boldsymbol{\beta}$ を設計している。$\boldsymbol{\beta}$ のノルムは 1 に正規化する。

\begin{align}
\boldsymbol{\beta} &\propto \left(\frac{1}{|S_+|}\sum_{v \in S_+} \vec{v} - \frac{1}{|S_\_|}\sum_{w \in S_\_} \vec{w} \right)
\end{align}

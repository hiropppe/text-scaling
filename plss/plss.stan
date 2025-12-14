
data {
  int<lower=1> D;                    // num docs
  int<lower=1> V;                    // num words
  int<lower=0> BoW[D, V];            // bag of words matrix (D x V)
  // array[D, V] int<lower=0> BoW;      // bag of words matrix (D x V) stan2.26+
  vector[V] p_v;                     // word prob
  vector[V] phi;                     // word polarity
}

parameters {
  vector[D] theta;
}

model {
  // prior: theta ~ N(0, 1)
  theta ~ normal(0, 1);

  // likelihood
  for (d in 1:D) {
    // log p(v|theta, phi) = log_softmax(log(p_v) + theta_d * phi)
    vector[V] log_probs = log_softmax(log(p_v) + theta[d] * phi);
    // log-likelihood: sum_v n_dv * log p(v|theta_d, phi)
    target += dot_product(to_vector(BoW[d]), log_probs);
  }
}

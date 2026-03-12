functions{
  // function that returns the log pdf of the wrapped-Cauchy
  real wrappedCauchy(real phi, real rho, real mu) {
    return(- log(2 * pi()) + log((1 - rho^2)/(1 + rho^2 - 2 * rho * cos(phi - mu))));
  }
  // returns exponential parameters if K==1
  real fGammaReturn(int K, real A){
    return(K == 1 ? A : 1.0);
  }
}

data {
  // base data
  int<lower=0> N;
  vector< upper=pi()>[N] turn;
  vector[N] dist;
  int<lower=1> K;
  int missing[N];
  
  // covariate
  int nCovs;
  matrix[N, nCovs + 1] X;
  
  // out-of-sample test
  int N1;
  vector<upper=pi()>[N1] turnTest;
  vector[N1] distTest;
  int missingTest[N1];
  matrix[N1, nCovs + 1] XTest;
}

parameters {
  // wrapped Cauchy params
  vector<lower=-pi(), upper=pi()>[K] mu;
  vector<lower=0,upper=1>[K] rho;
  
  // step parameters
  real<lower=1> a_step_1;
  real<lower=1,upper=3> b_step_1;
  real<lower=3> b_step_2;
  
  // switching parameters
  matrix[K * (K - 1), nCovs + 1] beta;
}

transformed parameters {
  matrix[K, K] Gamma_tr[N];
  vector[K] B_step;

  B_step[1] = b_step_1;
  B_step[2] = b_step_2;
  
  // determine switching probability matrix for each obs
  // since it depends on covariates
  {
  matrix[K, K] Gamma[N];
  matrix[K, K] Gamma1[N];
  
  for(n in 1:N){
    int a_count;
    a_count = 1;
    for(k_from in 1:K){
      for(k in 1:K){
        if(k_from == k){
          Gamma1[n, k_from, k] = 1;
        }else{
          Gamma1[n, k_from, k] = exp(beta[a_count] * to_vector(X[n]));
          a_count = a_count + 1;
        }
      }
      Gamma[n, k_from] = log(Gamma1[n, k_from] / sum(Gamma1[n, k_from]));
    }
  }
  // trace of switching prob matrix
  for(n in 1:N)
    for(k_from in 1:K)
      for(k  in 1:K)
        Gamma_tr[n, k, k_from] = Gamma[n, k_from, k];
  }
}

model {
  vector[K] lp;
  vector[K] lp_p1;
  
  lp = rep_vector(-log(K), K);
  
  // forwards algorithm
  for (n in 1:N) {
    // not missing
    if(missing[n] == 0){
      for (k in 1:K){
        lp_p1[k]
          = log_sum_exp(to_vector(Gamma_tr[n, k]) + lp)
            + wrappedCauchy(turn[n], rho[k], mu[k])
            + gamma_lpdf(dist[n] | fGammaReturn(k, a_step_1), B_step[k]);
      }
    }else{ // for missing just marginalise out previous state
      for (k in 1:K){
         lp_p1[k] = log_sum_exp(to_vector(Gamma_tr[n, k]) + lp);
      }
    }
    lp = lp_p1;
  }
  target += log_sum_exp(lp);
  
  a_step_1 ~ normal(1.5, 0.5);
  B_step[1] ~ normal(2, 0.5);
  B_step[2] ~ normal(3.5, 0.5);
  mu ~ normal(0, 0.5);
  rho[1] ~ normal(0.6, 0.1);
  rho[2] ~ normal(0.1, 0.1);
  for(i in 1:(K * (K - 1)))
    beta[i] ~ normal(0, 2);
}

generated quantities{
  
  // use Viterbi algorithm to determine optimal path
  int<lower=1,upper=K> state[N];
  real log_p_y_star;
  real lp_test_overall;
  vector[2] lp_test_element[N1];
  {
    int back_ptr[N, K];
    real best_logp[N, K];
    real best_total_logp;
    for (k in 1:K){
      if(missing[1] == 0){
        best_logp[1, K] = wrappedCauchy(turn[1], rho[k], mu[k])
                          + gamma_lpdf(dist[1] | fGammaReturn(k, a_step_1), B_step[k]);
      }else{
        best_logp[1, K] = 0;
      }
    }

    for (t in 2:N) {
      for (k in 1:K) {
      best_logp[t, k] = negative_infinity();
        for (j in 1:K) {
          real logp;
          if(missing[t] == 0){
            logp = best_logp[t - 1, j]
                   + Gamma_tr[t, k, j] + wrappedCauchy(turn[t], rho[k], mu[k])
                   + gamma_lpdf(dist[t] | fGammaReturn(k, a_step_1), B_step[k]);
          } else{
            logp = best_logp[t - 1, j] + Gamma_tr[t, k, j];
          }

          if (logp > best_logp[t, k]) {
            back_ptr[t, k] = j;
            best_logp[t, k] = logp;
          }
        }
      }
    }
    log_p_y_star = max(best_logp[N]);
    for (k in 1:K)
      if(best_logp[N, k] == log_p_y_star)
        state[N] = k;
      for (t in 1:(N - 1))
        state[N - t] = back_ptr[N - t + 1,
        state[N - t + 1]];
  }
  
  // for out-of-sample testing
  {
  matrix[N,K] B;
  vector[K] lp_p_total;
  vector[K] lp_p2;
  vector[K] lp_p1a;
  vector[K] lp_a;
  matrix[N,K] A;
  matrix[N,K] q_state;
  vector[N] p_state;
  matrix[N,K] p_state_all;

  // test sample
  vector[K] lp_test;
  vector[K] lp_test_temp;

  matrix[K,K] Gamma_test[N];
  matrix[K,K] Gamma_1_test[N];
  matrix[K,K] Gamma_tr_test[N];

  for(n in 1:N1){
    int a_count;
    a_count = 1;
    for(k_from in 1:K){
      for(k in 1:K){
        if(k_from == k){
          Gamma_1_test[n, k_from, k] = 1;
        }else{
          Gamma_1_test[n, k_from, k] = exp(beta[a_count] * to_vector(XTest[n]));
          a_count = a_count + 1;
        }
      }
      Gamma_test[n, k_from] = log(Gamma_1_test[n, k_from] / sum(Gamma_1_test[n, k_from]));
    }

  }

  for(n in 1:N1)
    for(k_from in 1:K)
      for(k  in 1:K)
        Gamma_tr_test[n, k, k_from] = Gamma_test[n, k_from, k];


  // forwards algorithm -- not Gamma_test here as it's for the sample not test
  lp_a = rep_vector(-log(K), K);

  for (n in 1:N) {
    if(missing[n] == 0){
        for (k in 1:K){
          lp_p1a[k] = log_sum_exp(to_vector(Gamma_tr[n, k]) + lp_a)
                      + wrappedCauchy(turn[n], rho[k], mu[k])
                      + gamma_lpdf(dist[n] | fGammaReturn(k, a_step_1), B_step[k]);
        }
      }else{
        for (k in 1:K){
          lp_p1a[k] = log_sum_exp(to_vector(Gamma_tr[n,k]) + lp_a);
        }
    }
    lp_a = lp_p1a;
    A[n] = to_row_vector(lp_a);
  }


  // backwards algorithm used here
  // http://www.cs.columbia.edu/~mcollins/fb.pdf
  // using forwards-backwards to obtain
  // p(state_1(t), state_2(t) | observations(1:t))
  B[N] = rep_row_vector(1, K);
  lp_p_total = rep_vector(0, K);

  for(n in 1:(N - 1)){
    if(missing[N - n + 1]==0){
       for(k in 1:K){
         lp_p2[k] = log_sum_exp(to_vector(Gamma_tr[n, k]) + lp_p_total)
                    + wrappedCauchy(turn[N - n + 1] , rho[k], mu[k])
                    + gamma_lpdf(dist[N - n + 1]| fGammaReturn(k, a_step_1), B_step[k]);
       }
    }else{
       for(k in 1:K){
         lp_p2[k] = log_sum_exp(to_vector(Gamma_tr[n, k]) + lp_p_total);
       }
    }
   lp_p_total = lp_p2;
   B[N - n] = to_row_vector(lp_p_total);
  }

  // estimate unnormalised state probs
  for(n in 1:N){
    for(k in 1:K){
      q_state[n, k] = A[n, k] + B[n, k];
    }
  }

  // normalise these to find prob of state 2 (using a 2 state model)
  for(n in 1:N){
      p_state[n] = exp(q_state[n, 2] - log_sum_exp(q_state[n]));
      p_state_all[n] = exp(q_state[n] - log_sum_exp(q_state[n]));
  }

  // out-of-sample predictive capability
  // use forward algorithm to calculate p(Y_1,Y_2,...,Y_k-1,X_k)
  lp_test = to_vector(p_state_all[N]);
  for (n in 1:N1) {
    if(missingTest[n] == 0){
      for (k in 1:K){
        lp_test_temp[k]
          = log_sum_exp(to_vector(Gamma_tr_test[n, k]) + lp_test)
            + wrappedCauchy(turnTest[n], rho[k], mu[k])
            + gamma_lpdf(distTest[n] | fGammaReturn(k, a_step_1), B_step[k]);
      }
    } else{
      for (k in 1:K){
        lp_test_temp[k]
          = log_sum_exp(to_vector(Gamma_tr_test[n, k]) + lp_test);
      }
    }

    lp_test = lp_test_temp;
    lp_test_element[n] = lp_test_temp;
  }
  // sum over X_k to marginalise
  lp_test_overall = log_sum_exp(lp_test);
  }
}

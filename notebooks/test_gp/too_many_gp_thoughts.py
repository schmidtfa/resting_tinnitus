with pm.Model(coords=coords) as glm_gp:

    #Priors for Regression part
    mu_a = pm.Normal('intercept', 0, 1)
    sigma_a = pm.HalfNormal('sigma_intercept', 2.5)
    alpha_i = pm.Normal('alpha_i|', 0, 1, dims="ch_name")
    

    mu_b = pm.Normal('beta', 0, 1)
    sigma_b = pm.HalfNormal('sigma_beta', 2.5)
    beta_i = pm.Normal('beta_i|', 0, 1, dims="ch_name")

    alpha = pm.Deterministic('1|', mu_a + alpha_i * sigma_a, dims="ch_name")
    beta = pm.Deterministic('beta|', mu_b + beta_i * sigma_b, dims="ch_name")

    # Priors for Gaussian Process (distance of parcels)
    eta_squared = pm.Exponential("eta_squared", 4)
    rho_squared = pm.Exponential("rho_squared", .5)

    #mean_func = pm.gp.mean.Linear(coeffs=beta[ch_ixs]*cur_df_cut['tinnitus'],
     #                             intercept=alpha[ch_ixs],
      #                            )
    
    #mean_func = pm.gp.mean.Linear(coeffs=0,
     #                             intercept=alpha,
      #                            )
    #mean_func = LinearModel(alpha=alpha[ch_ixs], beta=beta[ch_ixs])

    #mean_func = LinearModel(alpha[ch_ixs], beta[ch_ixs])
    #kernel_function = eta**2 * pm.gp.cov.ExpQuad(1, ls=rho)
    #latent = pm.gp.Latent(cov_func=kernel_function)
    #ch_eps = latent.prior("ch_eps", X=distance_matrix_2, dims="ch_name")

    cov_func = eta_squared * pm.gp.cov.ExpQuad(1, ls=rho_squared)
    gp = pm.gp.HSGP(m=[100], c=2, cov_func=cov_func)
    ch_eps = gp.prior("ch_eps", X=d_std, dims='ch_name')
    #gp = pm.gp.Marginal(mean_func=mean_func, cov_func=cov_func)
    #cov = gp.prior('cov', d_std, dims='ch_name')
    #cov = pm.Deterministic('cov', generate_L1_kernel_matrix(distance_matrix_2, eta, rho), dims='ch_name')
    #cov_func = eta * pm.gp.cov.Matern12(1, ls=rho)
    #likelihood

    # 
    # sigma = pm.Exponential('sigma', lam=1)
    # y = pm.Normal('y',
    #                mu=alpha[ch_ixs] + cur_df_cut['tinnitus'] + ch_eps[ch_ixs], #beta[ch_ixs] * 
    #                sigma=sigma,
    #                observed=zscore(cur_df_cut[feature]),
    #                dims='obs_id') 
    #sigma = pm.HalfNormal('sigma', 2.5)
    #K = pm.Deterministic('cov', np.eye(len(coords['ch_name'])) * sigma ** 2, dims='ch_name')
    # y = pm.MvNormal("y", 
    #                  mu=(alpha[ch_ixs] + beta[ch_ixs] * cur_df_cut['tinnitus']), 
    #                  cov=ch_eps[ch_ixs],
    #                  observed=zscore(cur_df_cut[feature]),
    #                  dims='obs_id'
    #                  )
    # Init the GP
    # sigma = pm.Exponential('sigma', lam=1)
    # gp.marginal_likelihood("y", 
    #                        X=d_std[ch_ixs], 
    #                        y=zscore(cur_df_cut[feature]), 
    #                        sigma=sigma,
    #                        dims='obs_id')

    #mdf = sample_numpyro_nuts(**sample_kwargs)
    #mdf =  pm.sample(**sample_kwargs)

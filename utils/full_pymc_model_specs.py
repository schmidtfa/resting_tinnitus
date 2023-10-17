    
#%%lin model    
    def _run_lin_reg(self, df, feature, sample_kwargs, non_centered=True, unpooled=True):

        cur_df = df[[feature, 'tinnitus_distress', 'ch_name']].dropna()

        ch_ixs, channel = pd.factorize(cur_df['ch_name'])
        coords = {
            "ch_name": channel,
            "obs_id": np.arange(len(ch_ixs)),
        }

        with pm.Model(coords=coords) as glm:

            if unpooled:
                #channel priors centered parametrization -> surprisingly faster than non-centered
                alpha = pm.Normal('1|', mu=0, sigma=1.5, dims="ch_name")
                beta = pm.Normal('beta|', mu=0, sigma=1, dims="ch_name")
            else:
                if non_centered:
                    mu_a = pm.Normal('intercept', 0, 1.5)
                    z_a = pm.Normal('z_a', 0, 1.5, dims="ch_name")
                    sigma_a = pm.Exponential('sigma_intercept', lam=1)


                    mu_b = pm.Normal('beta', 0, 1)
                    z_b = pm.Normal('z_b', 0, 1, dims="ch_name")
                    sigma_b = pm.Exponential('sigma_beta', lam=1)

                    #channel priors centered parametrization -> surprisingly faster than non-centered
                    alpha = pm.Deterministic('1|', mu_a + z_a * sigma_a, dims="ch_name")
                    beta = pm.Deterministic('beta|', mu_b + z_b * sigma_b, dims="ch_name")
                
                else:
                    #Hyperpriors
                    a = pm.Normal('intercept', 0, 1.5)
                    sigma_a = pm.Exponential('sigma_intercept', lam=1)
                    b = pm.Normal('beta', 0, 1)
                    sigma_b = pm.Exponential('sigma_beta', lam=1)

                    #channel priors centered parametrization -> surprisingly faster than non-centered
                    alpha = pm.Normal('1|', mu=a, sigma=sigma_a, dims="ch_name")
                    beta = pm.Normal('beta|', mu=b, sigma=sigma_b, dims="ch_name")

            #likelihood
            sigma = pm.Exponential('sigma',  lam=1)
            #psi = pm.Uniform('psi', 0.1, 0.9)
            observed = pm.HurdleLogNormal('tinnitus_distress',
                                          psi=pm.invlogit(alpha[ch_ixs] + beta[ch_ixs]*zscore(cur_df[feature])),
                                          mu=alpha[ch_ixs] + beta[ch_ixs]*zscore(cur_df[feature]),
                                          sigma=sigma,
                                          observed=cur_df['tinnitus_distress'],
                                          dims="obs_id")

            #mdf = sample_numpyro_nuts(**sample_kwargs)
            mdf =  pm.sample(**sample_kwargs)

        return mdf#, glm



#%% log model

    def _run_log_reg(self, df, feature, sample_kwargs, non_centered=True, bambi=False, unpooled=True):

        cur_df = df[[feature, 'tinnitus', 'ch_name']].dropna()


        if bambi:
            cur_df[feature] = zscore(cur_df[feature])
            import bambi as bmb
            md=bmb.Model(formula=f'tinnitus ~ 1 + {feature} + (1 + {feature}|ch_name)',
                         data=cur_df,
                         family='bernoulli',
                         )
            mdf = md.fit(**sample_kwargs)

        else:

            ch_ixs, channel = pd.factorize(cur_df['ch_name'])
            coords = {
                "ch_name": channel,
                "obs_id": np.arange(len(ch_ixs)),
            }

            with pm.Model(coords=coords) as glm:

                if unpooled:
                    alpha = pm.Normal('1|', mu=0, sigma=1.5, dims="ch_name")
                    beta = pm.Normal('beta|', mu=0, sigma=1, dims="ch_name")
                else:
                    if non_centered:
                        mu_a = pm.Normal('intercept', 0, 1.5)
                        z_a = pm.Normal('z_a', 0, 1.5, dims="ch_name")
                        sigma_a = pm.Exponential('sigma_intercept', lam=1)


                        mu_b = pm.Uniform('beta', lower=-3, upper=3)
                        z_b = pm.Normal('z_b', 0, 1, dims="ch_name")
                        sigma_b = pm.Exponential('sigma_beta', lam=1)

                        #channel priors centered parametrization -> surprisingly faster than non-centered
                        alpha = pm.Deterministic('1|', mu_a + z_a * sigma_a, dims="ch_name")
                        beta = pm.Deterministic('beta|', mu_b + z_b * sigma_b, dims="ch_name")
                    
                    else:
                        #Hyperpriors
                        a = pm.Normal('intercept', 0, 1.5)
                        sigma_a = pm.Exponential('sigma_intercept', lam=1)
                        b = pm.Normal('beta', 0, 1)
                        sigma_b = pm.Exponential('sigma_beta', lam=1)

                        #channel priors centered parametrization -> surprisingly faster than non-centered
                        alpha = pm.Normal('1|', mu=a, sigma=sigma_a, dims="ch_name")
                        beta = pm.Normal('beta|', mu=b, sigma=sigma_b, dims="ch_name")


                #likelihood
                observed = pm.Bernoulli('tinnitus',
                                        p=pm.math.invlogit(alpha[ch_ixs] + beta[ch_ixs]*zscore(cur_df[feature])),
                                        observed=cur_df['tinnitus'],
                                        dims="obs_id")

                #mdf = sample_numpyro_nuts(**sample_kwargs)
                mdf =  pm.sample(**sample_kwargs)

        return mdf#, glm
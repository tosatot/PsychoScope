import pymc as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('./model_outputs/bfi_data_formatted.csv')

# Step 1: Data Preparation
# Convert categorical variables to codes
data['model_code'] = data['model'].astype('category').cat.codes
data['persona_code'] = data['persona'].astype('category').cat.codes
data['trait_code'] = data['trait'].astype('category').cat.codes

# Step 2: Model Specification
with pm.Model() as hierarchical_model:
    # Hyperpriors
    # These are the overall mean and standard deviation for each group of effects
    mu_model = pm.Normal('mu_model', mu=0, sigma=10)
    sigma_model = pm.HalfNormal('sigma_model', sigma=5)
    
    mu_persona = pm.Normal('mu_persona', mu=0, sigma=10)
    sigma_persona = pm.HalfNormal('sigma_persona', sigma=5)
    
    mu_trait = pm.Normal('mu_trait', mu=0, sigma=10)
    sigma_trait = pm.HalfNormal('sigma_trait', sigma=5)
    
    # Model effects
    # This creates a normal distribution for each model's effect
    model_effects = pm.Normal('model_effects', mu=mu_model, sigma=sigma_model, 
                              shape=len(data['model'].unique()))
    
    # Persona effects (nested within models)
    # This creates a normal distribution for each persona's effect, for each model
    persona_effects = pm.Normal('persona_effects', mu=mu_persona, sigma=sigma_persona, 
                                shape=(len(data['model'].unique()), len(data['persona'].unique())))
    
    # Trait effects
    # This creates a normal distribution for each trait's effect
    trait_effects = pm.Normal('trait_effects', mu=mu_trait, sigma=sigma_trait, 
                              shape=len(data['trait'].unique()))
    
    # Expected value
    # This combines all the effects to predict the score
    mu = (model_effects[data['model_code']] + 
          persona_effects[data['model_code'], data['persona_code']] + 
          trait_effects[data['trait_code']])
    
    # Likelihood
    # This specifies how the observed scores are distributed around the expected value
    sigma = pm.HalfNormal('sigma', sigma=5)
    y = pm.Normal('y', mu=mu, sigma=sigma, observed=data['score'])

# Step 3: Model Fitting
with hierarchical_model:
    # Use the No-U-Turn Sampler (NUTS) to draw samples from the posterior
    # 'tune' specifies the number of samples to discard at the start
    # 'chains' specifies the number of independent sampling processes
    trace = pm.sample(2000, tune=1000, chains=4, cores=4)

# Step 4: Model Diagnostics
# Summary of the posterior distributions
summary = pm.summary(trace).round(2)
print("Model Summary:")
print(summary)

# Plot the traces of the samples
pm.plot_trace(trace)
plt.savefig('trace_plot.png')
plt.close()

# Check effective sample size and R-hat statistic
ess = pm.ess(trace)
rhat = pm.rhat(trace)
print("\nEffective Sample Size:")
print(ess)
print("\nR-hat Statistic:")
print(rhat)

# Step 5: Interpreting Results
# Plot posterior distributions
pm.plot_posterior(trace)
plt.savefig('posterior_plot.png')
plt.close()

# Compute highest density intervals (HDI)
hdi = pm.hdi(trace)
print("\nHighest Density Intervals:")
print(hdi)

# Compare effects
pm.plot_forest(trace, var_names=['model_effects', 'persona_effects', 'trait_effects'])
plt.savefig('forest_plot.png')
plt.close()

# Step 6: Model Comparison
# If you have multiple models, you can compare them using information criteria
# Compute WAIC (Widely Applicable Information Criterion)
waic = pm.waic(trace)
print("\nWidely Applicable Information Criterion (WAIC):")
print(waic)

# Compute LOO (Leave-One-Out Cross-Validation)
loo = pm.loo(trace)
print("\nLeave-One-Out Cross-Validation (LOO):")
print(loo)

# Step 7: Interpreting and Reporting Results
# Extract the posterior samples for further analysis
posterior_samples = trace.posterior

# Calculate the probability that larger models have higher BFI scores
larger_model_prob = (posterior_samples['model_effects'][:, :, -1] > 
                     posterior_samples['model_effects'][:, :, 0]).mean()

print("\nProbability that larger models have higher BFI scores:")
print(f"{larger_model_prob:.2%}")

# Calculate the effect sizes (difference between largest and smallest model)
effect_size = (posterior_samples['model_effects'][:, :, -1] - 
               posterior_samples['model_effects'][:, :, 0]).mean()

print("\nEstimated effect size (difference between largest and smallest model):")
print(f"{effect_size:.2f}")

# You can add more custom analyses here based on your specific research questions

print("\nInterpretation Guide:")
print("1. Look at the 'Model Summary' to see the estimated effects and their credible intervals.")
print("2. Check the trace plots to ensure the chains have mixed well.")
print("3. Examine the effective sample size (ESS) and R-hat statistic to verify convergence.")
print("4. Use the posterior plots and forest plots to visualize the distributions of effects.")
print("5. The HDI shows the 94% most probable values for each parameter.")
print("6. WAIC and LOO can be used to compare different models if you have multiple.")
print("7. The probability and effect size calculations give you direct answers to specific questions.")

print("\nWhen reporting, focus on describing the posterior distributions, credible intervals,")
print("and the practical significance of the effects. For example:")
print(f"'The probability that larger models have higher BFI scores is {larger_model_prob:.2%}.")
print(f"The estimated effect size (difference between largest and smallest model) is {effect_size:.2f}.")
print("This suggests a [positive/negative] relationship between model size and BFI scores.'")

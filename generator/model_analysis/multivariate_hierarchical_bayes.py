import pymc as pm
import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt

# Assuming you have a pandas DataFrame `data` with the structure:
# 'persona', 'model', 'size', 'Y1', 'Y2', 'Y3', 'Y4', 'Y5'
# Where 'persona', 'model', 'size' are categorical independent variables,
# and Y1 to Y5 are the dependent variables (subscales of the same instrument).

# Sample data structure (replace with actual data)
data = pd.DataFrame({
    'persona': np.random.choice([f'persona_{i+1}' for i in range(6)], size=100),
    'model': np.random.choice([f'model_{i+1}' for i in range(3)], size=100),
    'size': np.random.choice([f'size_{i+1}' for i in range(4)], size=100),
    'Y1': np.random.normal(size=100),
    'Y2': np.random.normal(size=100),
    'Y3': np.random.normal(size=100),
    'Y4': np.random.normal(size=100),
    'Y5': np.random.normal(size=100)
})

# Convert categorical variables to numeric indices
data['persona_code'] = pd.Categorical(data['persona']).codes
data['model_code'] = pd.Categorical(data['model']).codes
data['size_code'] = pd.Categorical(data['size']).codes

# Create design matrix for the independent variables (personas, models, sizes)
X = pd.get_dummies(data[['persona_code', 'model_code', 'size_code']], drop_first=True).values
n_samples, n_predictors = X.shape
n_responses = 5  # Number of dependent variables (subscales)

# Create the dependent variables matrix Y
Y = data[['Y1', 'Y2', 'Y3', 'Y4', 'Y5']].values

# Model Specification
with pm.Model() as hierarchical_model:
    
    # Priors for the intercepts (one for each response variable)
    intercept = pm.Normal("intercept", mu=0, sigma=10, shape=n_responses)
    
    # Priors for the regression coefficients (shared across dependent variables)
    beta = pm.Normal("beta", mu=0, sigma=1, shape=(n_predictors, n_responses))
    
    # Covariance matrix for the multivariate normal distribution of the dependent variables
    L_corr = pm.LKJCholeskyCov("L_corr", n=n_responses, eta=2.0, sd_dist=pm.HalfNormal.dist(1.0))
    cov = pm.Deterministic("cov", L_corr @ L_corr.T)  # Full covariance matrix
    
    # Expected value for the multivariate normal distribution
    mu = intercept + pm.math.dot(X, beta)

    # Multivariate normal likelihood for the observed subscale scores
    Y_obs = pm.MvNormal("Y_obs", mu=mu, chol=L_corr, observed=Y)

# Training (Sampling)
with hierarchical_model:
    trace = pm.sample(2000, return_inferencedata=True, target_accept=0.95)

# Posterior Predictive Checks (Validation)
with hierarchical_model:
    ppc = pm.sample_posterior_predictive(trace, var_names=["Y_obs"])

# Model Diagnostics
az.plot_trace(trace)
plt.savefig('trace.svg',format='svg')

az.summary(trace)

# Posterior Predictive Check Visualization
az.plot_ppc(az.from_pymc3(posterior_predictive=ppc, model=hierarchical_model), 
            data_pairs={"Y_obs": "Y_obs"})
plt.savefig('ppc.svg',format='svg')

# Posterior Predictive Check of Predicted vs Observed Data
predicted_mean = ppc['Y_obs'].mean(axis=0)

for i in range(n_responses):
    plt.scatter(Y[:, i], predicted_mean[:, i], label=f"Response {i+1}")
    plt.plot([Y[:, i].min(), Y[:, i].max()], [Y[:, i].min(), Y[:, i].max()], 'r--')
    plt.xlabel(f"Observed Y{i+1}")
    plt.ylabel(f"Predicted Y{i+1}")
    plt.legend()
    plt.title(f"Observed vs Predicted for Response {i+1}")
    plt.savefig(v'obs_vs_pred_{i}.svg',format='svg')

# Feature Selection and Shrinkage Diagnostics
# Check the posterior of the beta coefficients, looking at 95% credible intervals
az.plot_forest(trace, var_names=["beta"], combined=True, hdi_prob=0.95)
plt.title("Feature Importance (95% HDI for beta coefficients)")
plt.savefig('forest.svg',format='svg')

# Correlation Matrix Posterior Check
az.plot_posterior(trace, var_names=["cov"], hdi_prob=0.95)
plt.title("Posterior of the Covariance Matrix")
plt.savefig('posterior_covariance_matrix.svg',format='svg')


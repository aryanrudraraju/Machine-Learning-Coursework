import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.linear_model import Lasso, LassoCV, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# Imported to improve cleanliness of output
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# QUESTION 1 PART 1 - LOAD FILES

def load_data(path):
    # Helper function to read a natality CSV using NumPy
    return np.genfromtxt(
        path,
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8"
    )

# Load datasets for 2013, 2014, and 2015 (relative paths)
data2013 = load_data("natality2013ps.csv")
data2014 = load_data("natality2014ps.csv")
data2015 = load_data("natality2015ps.csv")



# QUESTION 1 PART 3 - KEEP SINGLETONS

def clean_year(data):
    """
    Restrict to singleton births and drop observations with
    missing or implausible values in key variables.
    """

    # 1) Singleton births only: DPLURAL = 1
    mask_singleton = (data["dplural"] == 1)

    # 2) Birthweight (grams): DBWT, keep 500–6000g
    bw = data["dbwt"].astype(float)
    mask_bw = (bw >= 500) & (bw <= 6000)

    # 3) Estimated gestational age (weeks): COMBGEST, keep 22–44 weeks
    ga = data["combgest"].astype(float)
    mask_ga = (ga >= 22) & (ga <= 44)

    # 4) Maternal smoking before pregnancy: CIG_0 (drop unknown 99 / negative)
    cig0 = data["cig_0"].astype(float)
    mask_cig0 = (cig0 >= 0) & (cig0 != 99)

    # 5) Maternal education: MEDUC, valid 1–8
    meduc = data["meduc"].astype(float)
    mask_meduc = (meduc >= 1) & (meduc <= 8)

    # 6) Child's year of birth: DOB_YY > 0
    year = data["dob_yy"].astype(float)
    mask_year = (year > 0)

    # Combine all filters
    mask_all = (
        mask_singleton &
        mask_bw &
        mask_ga &
        mask_cig0 &
        mask_meduc &
        mask_year 
    )

    return data[mask_all]

# Apply cleaning year by year
clean2013 = clean_year(data2013)
clean2014 = clean_year(data2014)
clean2015 = clean_year(data2015)

# Number of observations after cleaning in each year
print("Number of observations in each year after cleaning is \n", len(clean2013), len(clean2014), len(clean2015))



# QUESTION 1 PART 4 - DEFINE TREATMENT D

def define_treatment(data):
    """
    Treatment indicator D:
    D = 1 if mother smoked any cigarettes before pregnancy (CIG_0 > 0),
    D = 0 if CIG_0 == 0. Missing/unknown were already dropped in cleaning.
    """
    cig0 = data["cig_0"].astype(float)
    D = (cig0 > 0).astype(int)
    return D

D2013 = define_treatment(clean2013)
D2014 = define_treatment(clean2014)
D2015 = define_treatment(clean2015)

# Share of treated (smokers) in each year
print("The share of treated (smokers) in each year is \n", D2013.mean(), D2014.mean(), D2015.mean())

D_all = np.concatenate([D2013, D2014, D2015]) 

# Check number of elements in D_all
print("D_all has this many elements:", D_all.size)

# QUESTION 1 PART 5

# Combine all years (2013-2015) into single dataset
bw_all = np.concatenate([
    clean2013["dbwt"].astype(float),
    clean2014["dbwt"].astype(float),
    clean2015["dbwt"].astype(float)
])
# Combine MEDUC across years for later graphs
meduc_all = np.concatenate([
    clean2013["meduc"].astype(int),
    clean2014["meduc"].astype(int),
    clean2015["meduc"].astype(int)
])
# Combine COMBGEST across years for later graphs
ga_all = np.concatenate([
    clean2013["combgest"].astype(float),
    clean2014["combgest"].astype(float),
    clean2015["combgest"].astype(float)
])

def basic_summary(clean, D, year_label):
    """Print simple summaries for each year."""
    bw = clean["dbwt"].astype(float)
    ga = clean["combgest"].astype(float)

    n = len(bw)
    mean_bw = bw.mean()
    mean_ga = ga.mean()

    mean_bw_nosmoke = bw[D == 0].mean()
    mean_bw_smoke   = bw[D == 1].mean()

    print(f"\nYear {year_label}")
    print(f"Observations: {n}")
    print(f"Mean birthweight: {mean_bw:.1f} g")
    print(f"Mean gestational age: {mean_ga:.2f} weeks")
    print(f"Mean birthweight (non-smokers): {mean_bw_nosmoke:.1f} g")
    print(f"Mean birthweight (smokers):     {mean_bw_smoke:.1f} g")
    print(f"Smoking share D=1: {D.mean():.3f}")

def plot_bw_hist_density_combined(bw_all, D_all):
    """
    Density-normalised histograms of birthweight for smokers
    and non-smokers using combined 2013-2015 data, with KDE curves.
    Uses seaborn for KDE overlay.
    Density-normalised histograms of birthweight for smokers
    and non-smokers. Each group integrates to 1, so shapes are
    comparable even though group sizes differ a lot.
    """
    import seaborn as sns
    
    bw_n = bw_all[D_all == 0]
    bw_s = bw_all[D_all == 1]

    bins = np.linspace(500, 6000, 40)

    plt.figure(figsize=(10,5))
    
    # Plot histograms
    plt.hist(bw_n, bins=bins, density=True, alpha=0.6, label="Non-smokers (hist)")
    plt.hist(bw_s, bins=bins, density=True, alpha=0.6, label="Smokers (hist)")
    
    # Add KDE curves using seaborn
    sns.kdeplot(data=bw_n, color='darkblue', linewidth=2, label='Non-smokers (KDE)')
    sns.kdeplot(data=bw_s, color='darkred', linewidth=2, label='Smokers (KDE)')
    
    plt.xlabel("Birthweight (grams)")
    plt.ylabel("Density")
    plt.title("Birthweight distribution by smoking status\n(Combined 2013-2015)")
    plt.xlim(0, 6000)
    plt.legend()
    plt.tight_layout()
    plt.savefig("bw_hist_combined.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

def plot_gest_age_boxplot(ga_all, D_all):
    """
    Creates a comparative box plot of gestational age for smokers
    vs non-smokers to highlight differences in median, spread, and outliers.
    """

    ga_non = ga_all[D_all == 0]
    ga_smok = ga_all[D_all == 1]
    
    data = [ga_non, ga_smok]
    labels = ["Non-smokers", "Smokers"]
    colors = ['steelblue', 'darkorange'] # Blue and Orange

    plt.figure(figsize=(7, 6))
    
    boxplot_dict = plt.boxplot(
        data, 
        vert=True,            # Vertical box alignment
        patch_artist=True,    # Allows filling the boxes with color
        tick_labels=labels,        # Set x-axis labels
        medianprops={'color': 'black', 'linewidth': 2},
        capprops={'color': 'black', 'linewidth': 1.5},
        whiskerprops={'color': 'black', 'linestyle': '-', 'linewidth': 1.5},
        flierprops={'marker': 'o', 'markerfacecolor': 'red', 'markersize': 5, 'alpha': 0.5, 'markeredgecolor': 'none'}, # Outliers
        widths=0.6            # Control the width of the box
    )
    
    for patch, color in zip(boxplot_dict['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor('black')

    plt.title(
        "Comparative Box Plot of Gestational Age by Smoking Status",
        fontsize=14,
        fontweight='bold'
    )
    plt.xlabel("") # x-labels are set by the 'labels' argument in boxplot
    plt.ylabel("Gestational Age (weeks)", fontsize=12)
    
    plt.ylim(30, 45) 
    
    # Improve aesthetics
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='x', labelsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    
    plt.tight_layout()
    plt.savefig("gest_age_boxplot_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

def plot_smoking_by_education_combined(meduc_all, D_all):
    """
    Smoking rate by maternal education (combined 2013–2015),
    with descriptive x-axis labels for MEDUC codes 1–8.
    """

    # keep valid MEDUC codes 1–8 only
    mask_valid = (meduc_all >= 1) & (meduc_all <= 8)
    meduc = meduc_all[mask_valid]
    D = D_all[mask_valid]

    # MEDUC categories 1–8
    categories = np.arange(1, 9)

    # compute smoking rates per category
    rates = []
    for c in categories:
        m = (meduc == c)
        if m.sum() > 0:
            rates.append(D[m].mean())
        else:
            rates.append(np.nan)
    rates = np.array(rates)

    # nice short labels for the x-axis
    meduc_labels_short = [
        "1: ≤8th grade",
        "2: 9–12th\n grade,\nno diploma",
        "3: HS grad /\nGED",
        "4: Some\ncollege",
        "5: Assoc.\n(AA/AS)",
        "6: Bachelor\n(BA/BS)",
        "7: Master\n(MA/MS)",
        "8: Doctorate /\nProf."
    ]

    plt.figure(figsize=(10, 5))
    plt.bar(categories, rates, color="steelblue")
    plt.xticks(categories, meduc_labels_short, rotation=0, ha="center")
    plt.ylim(0, np.nanmax(rates) * 1.2)
    plt.xlabel("Maternal education (MEDUC)")
    plt.ylabel("Smoking rate (share with D = 1)")
    plt.title("Smoking rate by maternal education\n(Combined 2013–2015)")
    plt.tight_layout()
    plt.savefig("smoking_by_meduc_combined.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

# Create all combined plots
plot_smoking_by_education_combined(meduc_all, D_all)
plot_gest_age_boxplot(ga_all, D_all)
plot_bw_hist_density_combined(bw_all, D_all) 

# Create basic summary statistics 
for yr, clean, D in [(2013, clean2013, D2013),
                     (2014, clean2014, D2014),
                     (2015, clean2015, D2015)]:
    basic_summary(clean, D, yr)             



# QUESTION 2 PART 1 - DUMMIES FOR EDUCATION, YEAR OR BIRTH AND RESIDENCE

def make_dummies_one_col(data, colname):
    """
    Create dummy variables for one categorical column in the
    structured array 'data'. Drops the first category as baseline.
    Returns (dummy_matrix, dummy_names).
    """
    col = data[colname]
    categories = np.unique(col)          # sorted unique values
    baseline = categories[0]             # first category = baseline

    dummy_list = []
    name_list = []

    for c in categories[1:]:
        d = (col == c).astype(int)
        dummy_list.append(d)
        name_list.append(f"{colname}_{c}")

    # If there is only one category, return an empty matrix
    if len(dummy_list) == 0:
        return np.empty((len(data), 0), dtype=int), []

    return np.column_stack(dummy_list), name_list

def build_X_dummies(clean_data):
    """
    Build X from dummies for:
      - maternal education (meduc)
      - child year of birth (dob_yy)
    """
    X_parts = []
    names = []

    for col in ["meduc", "dob_yy"]:
        X_col, names_col = make_dummies_one_col(clean_data, col)
        X_parts.append(X_col)
        names.extend(names_col)

    X = np.column_stack(X_parts)
    return X, names

# Construct X for each year
X2013, Xnames = build_X_dummies(clean2013)
X2014, _      = build_X_dummies(clean2014)
X2015, _      = build_X_dummies(clean2015)

# Quick check: dimensions of the design matrices
print(X2013.shape, X2014.shape, X2015.shape)



# QUESTION 2 PART 2 - POLYNOMIALS TERMS OF MOTHER'S AGE AND GESTATIONAL AGE

def make_poly_features(data, colname, max_degree=3):
    """
    Create polynomial features x, x^2, ..., x^{max_degree} for
    the column `colname` in structured array `data`.
    Returns (poly_matrix, poly_names).
    """
    x = data[colname].astype(float)
    cols = []
    names = []

    for p in range(1, max_degree + 1):
        cols.append(x ** p)
        if p == 1:
            names.append(colname)
        else:
            names.append(f"{colname}{p}")   # e.g. mager2, mager3

    X_poly = np.column_stack(cols)
    return X_poly, names


def add_poly_to_X(clean_data, X_dum, names_dum):
    """
    Take existing dummy design matrix X_dum and append cubic
    polynomial terms for mager and combgest.
    """
    # mother’s age
    X_mage, names_mage = make_poly_features(clean_data, "mager", max_degree=3)
    # gestational age
    X_gest, names_gest = make_poly_features(clean_data, "combgest", max_degree=3)

    X_poly = np.column_stack([X_mage, X_gest])
    names_poly = names_mage + names_gest

    X_full = np.column_stack([X_dum, X_poly])
    names_full = names_dum + names_poly
    return X_full, names_full


# Extend X for each year: dummies + polynomials
X2013_full, Xnames_full = add_poly_to_X(clean2013, X2013, Xnames)
X2014_full, _           = add_poly_to_X(clean2014, X2014, Xnames)
X2015_full, _           = add_poly_to_X(clean2015, X2015, Xnames)

# Quick check of dimensions
print(X2013_full.shape, X2014_full.shape, X2015_full.shape)



# QUESTION 2 PART 3 - INTERACT MOTHER'S AGE AND EDUCATION DUMMIES

def add_age_education_interactions(clean_data, X_dum, names_dum):
    """
    Create interaction terms between mother's age (mager)
    and each maternal education dummy.
    """
    mage = clean_data["mager"].astype(float)

    # Extract education dummies created previously
    # X_dum should contain ALL dummy variables, including meduc dummies
    # We need to identify which dummy names correspond to meduc
    interaction_cols = []
    interaction_names = []

    for idx, name in enumerate(names_dum):
        if name.startswith("meduc_"):
            d = X_dum[:, idx]
            interaction = mage * d
            interaction_cols.append(interaction)
            interaction_names.append(f"{name}_x_mage")

    if len(interaction_cols) == 0:
        return np.empty((len(clean_data), 0)), []

    X_int = np.column_stack(interaction_cols)
    return X_int, interaction_names


# Apply interactions for each year
X2013_int, names2013_int = add_age_education_interactions(clean2013, X2013, Xnames)
X2014_int, names2014_int = add_age_education_interactions(clean2014, X2014, Xnames)
X2015_int, names2015_int = add_age_education_interactions(clean2015, X2015, Xnames)

# Add interactions onto the full X matrices (from 2.2)
X2013_final = np.column_stack([X2013_full, X2013_int])
X2014_final = np.column_stack([X2014_full, X2014_int])
X2015_final = np.column_stack([X2015_full, X2015_int])

# Track the names
Xnames_final = Xnames_full + names2013_int  # naming structure is same across years

# Quick check
print(X2013_final.shape, X2014_final.shape, X2015_final.shape)

X_all = np.vstack([X2013_final, X2014_final, X2015_final])

print(f"\nCombined dataset: {len(bw_all):,} observations, {X_all.shape[1]} features")




# QUESTION 3: LASSO FOR BIRTHWEIGHT PREDICTION (SMOKING UNPENALIZED)

print("QUESTION 3: LASSO FOR BIRTHWEIGHT PREDICTION")

# Standardise all covariates (not the smoking dummy)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_all)

# Scaling trick: scale smoking so its effective penalty is tiny
# A way to make the penalty negligible whilst still avoiding double LASSO (as it was not specified in the question)
# Elaborated more on in writeup
smoking_scale_factor = 1000  # make smoking penalty 1000x smaller 
D_scaled = D_all * smoking_scale_factor

# Design matrix for LASSO: [D_scaled, X_scaled]
X_with_smoking = np.column_stack([D_scaled, X_scaled])

# LASSO WITH 5-FOLD CV TO CHOOSE λ

lasso_cv = LassoCV(
    cv=5,
    max_iter=2000,
    tol=1e-4,
    n_jobs=-1,
    random_state=42
)

lasso_cv.fit(X_with_smoking, bw_all)
best_alpha = lasso_cv.alpha_
alpha_grid = lasso_cv.alphas_  # for plotting CV curve

# Extract coefficients
coef_smoking_scaled = lasso_cv.coef_[0]      # coefficient on scaled smoking
coef_covariates = lasso_cv.coef_[1:]         # coefficients on X_scaled

# Rescale smoking coefficient back to original scale
coef_smoking = coef_smoking_scaled * smoking_scale_factor

# Identify selected covariates (ignore numerically tiny ones)
selected_mask = np.abs(coef_covariates) > 1e-4
selected_indices = np.where(selected_mask)[0]
selected_coefs = coef_covariates[selected_mask]
n_selected = np.sum(selected_mask)


# RESULTS


print("RESULTS:")
print(f"Selected λ: {best_alpha:.6f}")
print(f"Variables: {n_selected + 1} (smoking + {n_selected} covariates)")
print(f"Smoking coefficient: {coef_smoking:.4f} grams")

if n_selected > 0:
    print("\nSelected covariates and coefficients (sorted by |coefficient|):")
    sort_idx = np.argsort(np.abs(selected_coefs))[::-1]
    for i in sort_idx:
        idx = selected_indices[i]
        print(f"  {Xnames_final[idx]}: {selected_coefs[i]:.4f}")

# VISUALISATIONS

# Figure 1: CV error vs λ

plt.figure(figsize=(7, 5))

mean_mse = lasso_cv.mse_path_.mean(axis=1)
std_mse = lasso_cv.mse_path_.std(axis=1)

plt.semilogx(alpha_grid, mean_mse, marker='o')
plt.fill_between(alpha_grid,
                 mean_mse - std_mse,
                 mean_mse + std_mse,
                 alpha=0.2)

plt.axvline(best_alpha, color='red', linestyle='--',
            label=fr'Selected λ = {best_alpha:.4f}')

plt.xlabel('λ (alpha)', fontsize=11)
plt.ylabel('Mean CV MSE', fontsize=11)
plt.title('5-fold Cross-Validation: Prediction Error vs λ (LASSO)',
          fontsize=12, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3, linestyle='--')
plt.tight_layout()
plt.show() 
plt.savefig('lasso_cv_mse_vs_lambda.png', dpi=300, bbox_inches='tight')
plt.close()


# Figure 2: Top selected features (bar plot)

plt.figure(figsize=(8, 6))
ax = plt.gca()

if n_selected > 0:
    # Sort by |coef|
    sort_idx = np.argsort(np.abs(selected_coefs))[::-1]

    # Show top 15
    top_n = min(15, len(sort_idx))
    top_coefs = selected_coefs[sort_idx[:top_n]]
    top_names = [Xnames_final[selected_indices[i]] for i in sort_idx[:top_n]]

    # Truncate very long names
    top_names = [n[:40] + '...' if len(n) > 40 else n for n in top_names]

    # Colour by sign
    colors = ['steelblue' if c > 0 else 'indianred' for c in top_coefs]
    y_pos = np.arange(len(top_coefs))

    ax.barh(y_pos, top_coefs, color=colors, alpha=0.7,
            edgecolor='black', linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_names, fontsize=9)
    ax.set_xlabel('Coefficient (grams)', fontsize=11)
    ax.set_title(f'Top {top_n} Selected Covariates\n(LASSO with Smoking Unpenalized)',
                 fontsize=12, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    legend_elements = [
        Patch(facecolor='steelblue', alpha=0.7, label='Positive effect'),
        Patch(facecolor='indianred', alpha=0.7, label='Negative effect')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
else:
    ax.text(0.5, 0.5, 'No covariates selected', ha='center', va='center',
            transform=ax.transAxes, fontsize=12)
    ax.set_title('Top Selected Covariates (LASSO)', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()  
plt.savefig('lasso_top_selected_covariates.png', dpi=300, bbox_inches='tight')
plt.close()





# QUESTION 4 - MECHANICAL DOUBLE LASSO PROCEDURE

# bw: birthweight outcome variable
bw = bw_all

# QUESTION 4 PART 1 -  LASSO regression of birthweight on D and X

# Create penalty factor: 0 for D (no penalty), 1 for all covariates
# Scale D by a large number so its coefficient is effectively unpenalized
D_scaled = D_all * 1e6
print("D_scaled is", D_scaled.size)
print("X_all:",X_all.size)
X_step1 = np.column_stack([D_scaled, X_all])

# Run LASSO with 5-fold CV
lasso_y = LassoCV(cv=5, random_state=42, max_iter=10000)
lasso_y.fit(X_step1, bw)

# Identify variables with nonzero coefficients (excluding D)
# Skip first coefficient (D) and check remaining
coef_y = lasso_y.coef_
nonzero_idx_y = np.where(np.abs(coef_y[1:]) > 1e-10)[0]
S_Y = nonzero_idx_y
print(f" Number of selected covariates in S_Y: {len(S_Y)}")



# QUESTION 4 PART 2 - LASSO regression of D on X

lasso_d = LassoCV(cv=5, random_state=42, max_iter=10000)
lasso_d.fit(X_all, D_all)

# Identify variables with nonzero coefficients
coef_d = lasso_d.coef_
nonzero_idx_d = np.where(np.abs(coef_d) > 1e-10)[0]
S_D = nonzero_idx_d
print(f"Number of selected covariates in S_D: {len(S_D)}")



# QUESTION 4 PART 3 - Form the union of selected controls

S_union = np.union1d(S_Y, S_D)
print(f"\nNumber of covariates in S (union): {len(S_union)}")



# QUESTION 4 PART 4 - OLS regression

# Extract selected covariates
X_selected = X_all[:, S_union]

# Create design matrix for OLS: D and selected X
X_ols = np.column_stack([D_all, X_selected])

# Fit OLS
ols = LinearRegression()
ols.fit(X_ols, bw)

# Extract treatment effect 
tau_hat = ols.coef_[0]

# Calculate standard error
# Residuals
residuals = bw - ols.predict(X_ols)
n = len(bw)
k = X_ols.shape[1]

# Calculate variance-covariance matrix
residual_var = np.sum(residuals**2) / (n - k)
XtX_inv = np.linalg.inv(X_ols.T @ X_ols)
var_covar = residual_var * XtX_inv

# Standard error for tau 
se_tau = np.sqrt(var_covar[0, 0])

print(f"\nStep 4 - OLS Results:")
print(f"Estimated treatment effect (tau_hat): {tau_hat:.4f}")
print(f"Standard error: {se_tau:.4f}")
print(f"95% Confidence Interval: [{tau_hat - 1.96*se_tau:.4f}, {tau_hat + 1.96*se_tau:.4f}]")



# QUESTION 4 PART 6 - Visualization of the result

# Coefficient plot showing estimated effect with confidence interval
fig, ax = plt.subplots(figsize=(8, 5))

ax.errorbar(0, tau_hat, yerr=1.96*se_tau, fmt='o', markersize=10, 
            capsize=10, capthick=2, color='darkblue', elinewidth=2)
ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.7)
ax.set_xlim(-0.5, 0.5)
ax.set_xticks([0])
ax.set_xticklabels(['Maternal Smoking\nBefore Pregnancy'])
ax.set_ylabel('Effect on Birthweight (grams)', fontsize=12)
ax.set_title('Double LASSO: Estimated Causal Effect of Maternal Smoking on Birthweight', 
             fontsize=13, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# Add text box with estimate
textstr = f'Effect: {tau_hat:.2f} grams\nSE: {se_tau:.2f}\n95% CI: [{tau_hat - 1.96*se_tau:.2f}, {tau_hat + 1.96*se_tau:.2f}]'
ax.text(0.02, 0.98,
        textstr, 
        transform=ax.transAxes, 
        fontsize=11,
        verticalalignment='top', 
        bbox=dict(boxstyle='round', facecolor='white', alpha=1),
        )

plt.tight_layout()
plt.savefig('double_lasso_effect.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nDouble LASSO procedure completed. Figures saved.")

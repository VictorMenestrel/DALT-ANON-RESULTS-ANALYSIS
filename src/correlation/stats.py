"""Linear regression of human on ASR intelligibility and residual diagnostics."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import linregress, pearsonr, shapiro, t
from statsmodels.stats.diagnostic import het_breuschpagan

from .config import ALPHA, X_COLUMN, Y_COLUMN


@dataclass
class RegressionResult:
    """Linear regression of human intelligibility on ASR intelligibility."""

    correlation: float
    slope: float
    intercept: float
    p_value: float
    std_err: float
    residuals: pd.Series

    @property
    def r_squared(self) -> float:
        """Coefficient of determination."""
        return self.correlation**2


def fit_regression(merged_df: pd.DataFrame) -> RegressionResult:
    """Fit the linear regression of human intelligibility on ASR intelligibility.

    Args:
        merged_df: Output of :func:`correlation.data.get_merged_df`.

    Returns:
        Regression coefficients, statistics and residuals.
    """
    x, y = merged_df[X_COLUMN], merged_df[Y_COLUMN]

    correlation = x.corr(y)
    # pyrefly: ignore [bad-unpacking]
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    assert abs(r_value - correlation) < 1e-6, "R values do not match!"

    return RegressionResult(
        correlation=correlation,
        slope=slope,
        intercept=intercept,
        p_value=p_value,
        std_err=std_err,
        residuals=y - (slope * x + intercept),
    )


def report_regression(merged_df: pd.DataFrame, result: RegressionResult) -> None:
    """Print the regression statistics and the residual diagnostics.

    The residuals are tested for heteroscedasticity (Breusch-Pagan) and
    normality (Shapiro-Wilk). The RMSE is given both against the identity line
    (raw ASR scores) and against the regression line (RMSE*).

    Args:
        merged_df: Output of :func:`correlation.data.get_merged_df`.
        result: Output of :func:`fit_regression`.
    """
    residuals = result.residuals

    print(f"Correlation (R): {result.correlation}")
    print(f"R-squared: {result.r_squared}")

    X = sm.add_constant(merged_df[X_COLUMN])
    lm_statistic, lm_p_value, f_statistic, f_p_value = het_breuschpagan(residuals, X)
    print(f"Breusch-Pagan test LM: {lm_statistic:.4f}, p-value: {lm_p_value:.4f}")
    print(f"Breusch-Pagan test F: {f_statistic:.4f}, p-value: {f_p_value:.4f}")
    if f_p_value > ALPHA:
        print(
            "Cool! Fail to reject the null hypothesis: No heteroscedasticity detected."
        )
    else:
        print("Warning! Reject the null hypothesis: Heteroscedasticity detected.")

    print(f"Regression line: y = {result.slope:.6f}x + {result.intercept:.6f}")
    print(f"p-value: {result.p_value:.4f}")
    print(f"Standard error: {result.std_err:.4f}")
    print(f"Scipy linregress R-squared: {result.r_squared:.4f}")
    print(f"Mean of residuals (0 is expected): {residuals.mean():.4f}")

    rmse = (merged_df[Y_COLUMN] - merged_df[X_COLUMN]).pow(2).mean() ** 0.5
    print(f"RMSE:  {rmse:.4f}")
    rmse_star = residuals.pow(2).mean() ** 0.5
    print(f"RMSE*: {rmse_star:.4f}")

    shapiro_statistic, shapiro_p_value = shapiro(residuals)
    print(
        f"Shapiro-Wilk test statistic: {shapiro_statistic:.4f}, p-value: {shapiro_p_value:.4f}"
    )
    if shapiro_p_value > ALPHA:
        print(
            "Cool! Fail to reject the null hypothesis: Residuals are normally distributed."
        )
    else:
        print(
            "Warning! Reject the null hypothesis: Residuals are not normally distributed."
        )


@dataclass
class SteigerResult:
    """Steiger-Williams test comparing two dependent correlations with the same target."""

    r_1_target: float
    r_2_target: float
    r_1_2: float
    n: int
    t_statistic: float
    p_value: float


def steiger_williams_test(
    r12: float, r13: float, r23: float, n: int
) -> tuple[float, float]:
    """Test the difference between two dependent correlations sharing one variable.

    Williams' t, as recommended by Steiger (1980), tests H0: r12 == r13, where
    variable 1 is the shared target and variables 2 and 3 are the two predictors.

    Args:
        r12: Correlation between the target and predictor 1.
        r13: Correlation between the target and predictor 2.
        r23: Correlation between the two predictors.
        n: Number of observations.

    Returns:
        The t statistic (n - 3 degrees of freedom) and its two-sided p-value.
    """
    # Determinant of the 3x3 correlation matrix, clipped to avoid a negative
    # value due to rounding when the predictors are almost identical
    R_det = max(1 - r12**2 - r13**2 - r23**2 + 2 * r12 * r13 * r23, 0)

    t_statistic = (r12 - r13) * np.sqrt((n - 3) * (1 + r23)) / np.sqrt(2 * R_det)
    p_value = 2 * (1 - t.cdf(abs(t_statistic), n - 3))
    return t_statistic, p_value


def compare_correlations(
    df: pd.DataFrame, predictor_1: str, predictor_2: str, target: str = Y_COLUMN
) -> SteigerResult:
    """Compare the correlations of two predictors with the same target.

    Args:
        df: Table with one column per predictor and the target column.
        predictor_1: Column of the first predictor.
        predictor_2: Column of the second predictor.
        target: Column of the target.

    Returns:
        The three correlations and the Steiger-Williams test result.
    """
    r_1_target, _ = pearsonr(df[predictor_1], df[target])
    r_2_target, _ = pearsonr(df[predictor_2], df[target])
    r_1_2, _ = pearsonr(df[predictor_1], df[predictor_2])
    n = len(df)

    # pyrefly: ignore [bad-argument-type]
    t_statistic, p_value = steiger_williams_test(r_1_target, r_2_target, r_1_2, n)

    return SteigerResult(
        # pyrefly: ignore [bad-argument-type]
        r_1_target=r_1_target,
        # pyrefly: ignore [bad-argument-type]
        r_2_target=r_2_target,
        # pyrefly: ignore [bad-argument-type]
        r_1_2=r_1_2,
        n=n,
        t_statistic=t_statistic,
        p_value=p_value,
    )


def report_steiger(result: SteigerResult, name_1: str, name_2: str) -> None:
    """Print the correlations and the Steiger-Williams test result.

    Args:
        result: Output of :func:`compare_correlations`.
        name_1: Name of the first predictor.
        name_2: Name of the second predictor.
    """
    print(f"r({name_1}, Human) : {result.r_1_target:.4f}")
    print(f"r({name_2}, Human) : {result.r_2_target:.4f}")
    print(f"r({name_1}, {name_2}) : {result.r_1_2:.4f}")
    print(f"N : {result.n}")
    print("-" * 40)
    print("Steiger-Williams test for dependent correlations:")
    print(f"t-statistic : {result.t_statistic:.4f}")
    print(f"p-value     : {result.p_value:.4f}")
    if result.p_value > ALPHA:
        print("Fail to reject the null hypothesis: no significant difference.")
    else:
        print("Reject the null hypothesis: the correlations differ significantly.")

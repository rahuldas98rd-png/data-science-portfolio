import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st

from pathlib import Path

# Get project root
PROJECT_ROOT = Path.cwd().parent
# Figure directory
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"


def histogrammer(df: pd.DataFrame, 
                 column: str, 
                 median_text: bool=True, 
                 mean_text: bool=True,
                 save_fig: bool=False,
                 **kwargs):
    """
    Helper function to plot histograms based on the data type of the column. 
    It also calculates and displays the median, mean, skewness, and kurtosis 
    for numerical columns, and the mode for categorical columns.

    **kwargs = any keyword arguments from the sns.histplot() function
    """
    
    plt.figure(figsize=(15,5))

    if 'hue' in kwargs:
        median_text = False
        mean_text = False

    if (df[column].dtypes != 'str') and (df[column].dtypes != 'object'):
        median=round(df[column].median(), 1)
        mean=round(df[column].mean(), 1)
        skewness=round(df[column].skew(), 1)
        kurtosis=round(st.kurtosis(df[column], fisher=False), 1)

        ax = sns.histplot(x=df[column], **kwargs)           # Plot the histogram
        plt.axvline(median, color='red', linestyle='--')    # Plot the median line
        plt.axvline(mean, color='green', linestyle='--')
        
        if (median_text==True) and (mean_text==True):       # Add median text unless set to False
            print("Skewness: {}\nKurtosis: {}".format(skewness, kurtosis))
            ax.text(0.25, 0.85, f'median={median}', color='red',
                ha='left', va='top', transform=ax.transAxes)
            ax.text(0.25, 0.95, f'mean={mean}', color='green',
                ha='left', va='top', transform=ax.transAxes)
        else:
            print('Median:', median)
            print('Mean:', mean)
    
    else:
        mode = df[column].mode()[0]
        print('Mode:', mode)
        sns.histplot(x=df[column], **kwargs)
        plt.xticks(rotation=90)
    plt.title(f'{column} histogram')
    if save_fig:
        plt.savefig(FIGURE_DIR / f'{column}_histogram.png', dpi=300, bbox_inches='tight')
    plt.show()


def detect_outliers(df: pd.DataFrame,
                    columns: list):
    """Detect outliers in a numerical column using the IQR method."""
    
    for col in columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        n_low  = (df[col] < lower).sum()
        n_high = (df[col] > upper).sum()
        print(f"{col:20s} | IQR=[{q1:.2f}, {q3:.2f}] | bounds=[{lower:.2f}, {upper:.2f}] | low={n_low} | high={n_high}")
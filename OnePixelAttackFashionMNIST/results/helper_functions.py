import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np


sns.set_theme(style="whitegrid")
palette = sns.color_palette("hls", 6)


def conf_matrix(df, model_name, pixel_count):
    """
    Generates a confusion matrix for a specific model and pixel count.
    """
    conf_matrix = confusion_matrix(df['actual_class'], df['predicted_class'], labels=df['actual_class'].unique())
    return conf_matrix


def plot_confusion_matrices_by_model(model_tables, conf_matrix_set, pixel_counts=[1, 2, 3, 4]):
    """
    Plots confusion matrices for each model with subplots for each pixel count (2x2 grid).
    """
    palette = sns.color_palette("hls", len(pixel_counts))
    for model in model_tables.keys():
        fig, axes = plt.subplots(2, 2, figsize=(12, 10)) 
        fig.suptitle(f"Confusion Matrices for {model}", fontsize=16)

        for i, pixel in enumerate(pixel_counts):
            ax = axes[i // 2, i % 2]  
            matrix_key = f"{model} {pixel} Pixel"

            if matrix_key in conf_matrix_set:
                conf_matrix_data = conf_matrix_set[matrix_key]
                sns.heatmap(conf_matrix_data, annot=True, fmt="d",
                            cmap=sns.light_palette(palette[i], as_cmap=True),
                            xticklabels=model_tables[model].index,
                            yticklabels=model_tables[model].index,
                            ax=ax)
                ax.set_title(f"{pixel} Pixel Attack")
                ax.set_xlabel("Predicted Class")
                ax.set_ylabel("Actual Class")
            else:
                ax.axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()


def success_by_class(df, targeted = False):
  """
    Calculates the success rate for each class based on the 'success' column.

    Parameters:
    - df: DataFrame containing the data. Must include 'actual_class' and 'success' columns.

    Returns:
    - success_rate_per_class: A pandas Series where the index is 'actual_class' and the value is
      the success rate as a formatted string with two decimal places and a '%' sign.
  """
  if targeted:
      success_rate_per_class = df.groupby('actual_class')['targeted_success'].mean() * 100
  else:
      success_rate_per_class = df.groupby('actual_class')['success'].mean() * 100
  success_rate_per_class = success_rate_per_class.apply(lambda x: f"{x:.2f}%")
  return success_rate_per_class

def plot_success_rate_by_class(model_tables, pixel_counts=[1, 2, 3, 4, 5, 6]):
    """
    Plots line charts for success rates by class for each model and pixel count,
    using a consistent color palette.

    Parameters:
    - model_tables: Dictionary of DataFrames, where keys are model names and values are DataFrames with success rates.
    - pixel_counts: List of pixel counts to consider for each model.
    """
    palette = sns.color_palette("hls", len(pixel_counts))  # Define a consistent color palette

    for model, df in model_tables.items():
        # Convert percentage strings to floats
        model_data = df.applymap(lambda x: float(x.strip('%')))

        # Plotting
        fig, ax = plt.subplots(figsize=(10, 6))

        for i, pixel in enumerate(pixel_counts[:len(model_data.columns)]):
            ax.plot(
                model_data.index,  # Class labels (assumes index contains class labels)
                model_data.iloc[:, i],  # Success rates for the pixel count
                marker='o',  # Add markers for clarity
                label=f"{pixel} Pixel",
                color=palette[i]  # Use palette color
            )

        # Chart customization
        ax.set_title(f"Success Rate by Class for {model}", color='black')  # Optional: Change title color
        ax.set_xlabel("Class Label")
        ax.set_ylabel("Success Rate (%)")
        ax.legend(title="Pixel Count")
        ax.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(rotation=0)  # Keep class labels horizontal
        plt.show()

def plot_success_rate_by_pixel_count(df):
    """
    Plots a grouped bar chart for attack success rate by model and pixel count.

    Parameters:
    - df: DataFrame with 'model_name', 'pixel_count', and 'success' columns.
    """

    success_by_pixel_count = df.groupby(['model_name', 'pixel_count'])['success'].mean().unstack() * 100
    success_by_pixel_count.plot(kind='bar', stacked=False, color=palette, figsize=(10, 6))
    plt.xlabel('Model Name')
    plt.ylabel('Attack Success Rate (%)')
    plt.title('Attack Success Rate by Model and Pixel Count')
    plt.legend(title="Pixel Count")
    plt.xticks(rotation=0)
    plt.show()

def get_success_rate_table_by_pixel_count(df):
    """
    Returns a summary table of attack success rate by model and pixel count.

    Parameters:
    - df: DataFrame with 'model_name', 'pixel_count', and 'success' columns.

    Returns:
    - DataFrame: A summary table with models as rows, pixel counts as columns, and success rates as values.
    """
    success_by_pixel_count = df.groupby(['model_name', 'pixel_count'])['success'].mean().unstack() * 100
    success_by_pixel_count = success_by_pixel_count.round(2)
    success_by_pixel_count = success_by_pixel_count.applymap(lambda x: f"{x:.2f}%")
    return success_by_pixel_count

def confidence_diff_boxplot(df, model_names, pixel_counts):
    """
    Plots a box plot of confidence difference by attack outcome (Successful vs. Unsuccessful),
    for each model and pixel count combination.

    Parameters:
    - df: DataFrame containing 'success', 'confidence_diff', 'model_name', and 'pixel_count' columns.
    - model_names: List of model names (e.g., ['VGG16', 'ResNet']).
    - pixel_counts: List of pixel counts (e.g., [1, 3, 5]).
    """

    df['attack_outcome'] = df['success'].replace({1: 'Successful', 0: 'Unsuccessful'})
    plot_data = df[df['model_name'].isin(model_names) & df['pixel_count'].isin(pixel_counts)]

    for model in model_names:  # Plot separately for each model
        model_data = plot_data[plot_data['model_name'] == model]
        plt.figure(figsize=(12, 8))
        sns.boxplot(data=model_data,
                    x='attack_outcome',
                    y='confidence_diff',
                    hue='pixel_count',
                    palette=sns.color_palette("hls", len(pixel_counts)))

        # Construct the dynamic title
        plt.xlabel('Attack Outcome')
        plt.ylabel('Confidence Difference')
        plt.title(f'Confidence Difference for Successful vs Unsuccessful Attacks\n'
                  f'{model}')
        plt.legend(title='Pixel Count', loc='upper left')
        plt.show()

def identify_outliers(df, model_names, pixel_counts):
    """
    Identifies outliers in the dataset based on the confidence difference for each attack outcome
    and pixel count combination, and returns the rows corresponding to the outliers.

    Parameters:
    - df: DataFrame containing 'success', 'confidence_diff', 'model_name', and 'pixel_count' columns.
    - model_names: List of model names (e.g., ['VGG16', 'ResNet']).
    - pixel_counts: List of pixel counts (e.g., [1, 3, 5]).

    Returns:
    - outliers_df: DataFrame containing the rows of the outliers.
    """
    # Add a column for the attack outcome
    df['attack_outcome'] = df['success'].replace({1: 'Successful', 0: 'Unsuccessful'})

    # Filter the dataframe for the selected model names and pixel counts
    df_filtered = df[df['model_name'].isin(model_names) & df['pixel_count'].isin(pixel_counts)]

    outliers = []

    # Loop through each combination of attack outcome and pixel count
    for attack_outcome in df_filtered['attack_outcome'].unique():
        for pixel_count in df_filtered['pixel_count'].unique():
            # Get the subset of data for the current combination
            subset = df_filtered[(df_filtered['attack_outcome'] == attack_outcome) &
                                 (df_filtered['pixel_count'] == pixel_count)]

            # Calculate the IQR (Interquartile Range)
            Q1 = subset['confidence_diff'].quantile(0.25)
            Q3 = subset['confidence_diff'].quantile(0.75)
            IQR = Q3 - Q1

            # Calculate the lower and upper bounds for outliers
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Identify outliers (rows where confidence_diff is outside the bounds)
            outlier_rows = subset[(subset['confidence_diff'] < lower_bound) |
                                  (subset['confidence_diff'] > upper_bound)]

            # Append the outlier rows to the list
            outliers.append(outlier_rows)

    # Combine all outliers into a single DataFrame
    outliers_df = pd.concat(outliers) if outliers else pd.DataFrame()

    return outliers_df

def plot_outliers(outliers_df):
    """
    Plots the number of outliers by model and by pixel count side by side.
    
    Parameters:
    - outliers_df (pd.DataFrame): dataframe with accumulated outliers data.
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 6), constrained_layout=True)

    # Count of outliers by model
    outliers_by_model = outliers_df['model_name'].value_counts()

    # Count of outliers by pixel count
    outliers_by_pixel = outliers_df['pixel_count'].value_counts().sort_index()
    
    # Bar plot for models
    sns.barplot(ax=axes[0], x=outliers_by_model.index, y=outliers_by_model.values, palette=palette)
    axes[0].set_title("Number of Outliers by Model")
    axes[0].set_xlabel("Model Name")
    axes[0].set_ylabel("Count of Outliers")
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(False)

    # Add numbers above bars
    for i, value in enumerate(outliers_by_model.values):
        axes[0].text(i, value + 1,  
                str(value),    
                ha='center',   
                va='bottom',  
                fontsize=10)   
    
    # Bar plot for pixel counts
    sns.barplot(ax=axes[1], x=outliers_by_pixel.index, y=outliers_by_pixel.values, palette=palette)
    axes[1].set_title("Number of Outliers by Pixel Count")
    axes[1].set_xlabel("Pixel Count")
    axes[1].set_ylabel("Count of Outliers")
    axes[1].grid(False)

    # Add numbers above bars
    for i, value in enumerate(outliers_by_pixel.values):
        axes[1].text(i, value + 1, 
                str(value),    
                ha='center',  
                va='bottom',   
                fontsize=10)   
plt.show()


def outliers_by_class_label(outliers_df):
    """
    Plots the class distribution for each model in the given DataFrame.

    Parameters:
    - outliers_df (pd.DataFrame): DataFrame containing the data with columns 'model_name' and 'actual_class'.

    Returns:
    - None: Displays the plots.
    """
    # Group by model_name and actual_class to count occurrences
    class_distribution = outliers_df.groupby(["model_name", "actual_class"]).size().reset_index(name="count")

    # Get unique models
    models = outliers_df["model_name"].unique()

    # Set up the plots
    fig, axes = plt.subplots(1, len(models), figsize=(15, 5), sharey=True)

    # Set the color palette
    palette = sns.color_palette("hls", 10)

    # Generate plots for each model
    for i, model in enumerate(models):
        ax = axes[i]
        model_data = class_distribution[class_distribution["model_name"] == model]
        bars = ax.bar(
            model_data["actual_class"],
            model_data["count"],
            color=palette[5]
        )
        ax.set_title(f"Outliers by Class Label for {model}")
        ax.set_xlabel("True Class")
        ax.set_ylabel("Count")
        ax.set_xticks(model_data["actual_class"])
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        ax.grid(False)
        ax.set_ylim(0, 100)

        # Add numbers above bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2, 
                height + 1,                       
                str(height),                       
                ha="center",                      
                va="bottom",                       
                fontsize=9                         
            )

    plt.tight_layout()
    plt.show()



# Function to clean the string and convert it into a numpy array
def convert_to_array(prob_str):
    # Use numpy.fromstring to convert the cleaned string into a numpy array
    clean_str = prob_str.strip("[]")
    return np.fromstring(clean_str, sep=' ')


# Function to safely get the predicted probability
def get_pred_conf(row):
    # Check if the predicted_class is within the bounds of predicted_probs
    if row['predicted_class'] < len(row['predicted_probs']):
        return row['predicted_probs'][row['predicted_class']]
    else:
        print("something wrong")
        return None
    
def calculate_confidence_means(df):
    """
    Calculates the average pred_conf for each combination of model_name and pixel_count
    where success is True and formats the result to 5 significant figures with 2 decimal places.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing columns 'success', 'pred_conf', 'model_name', and 'pixel_count'.

    Returns:
    pd.DataFrame: A DataFrame with columns 'model_name', 'pixel_count', and the average 'pred_conf'.
    """
    df_success = df[df['success'] == True]

    # Group by 'model_name' and 'pixel_count' and calculate the mean of 'pred_conf'
    mean_confidence = df_success.groupby(['model_name', 'pixel_count'])['pred_conf'].mean().reset_index()

    # Convert 'pred_conf' to percentage and round to 2 decimal places
    mean_confidence['pred_conf'] = (mean_confidence['pred_conf'] * 100).round(2).astype(str) + '%'

    # Pivot the table: model_name as rows, pixel_count as columns, and pred_conf as values
    pivot_confidence = mean_confidence.pivot(index='model_name', columns='pixel_count', values='pred_conf')

    return pivot_confidence


################################################################################################################
################################################################################################################


def get_targeted_success_rate_table_by_pixel_count(df):
    """
    Returns a summary table of targeted attack success rate by model and pixel count.

    Parameters:
    - df: DataFrame with 'model_name', 'pixel_count', 'target_class', and 'predicted_class' columns.

    Returns:
    - DataFrame: A summary table with models as rows, pixel counts as columns, and success rates as values.
    """
    # Define success for targeted attacks
    df['targeted_success'] = df['target_class'] == df['predicted_class']

    # Calculate success rate
    success_by_pixel_count = df.groupby(['model_name', 'pixel_count'])['targeted_success'].mean().unstack() * 100

    # Format and round values
    success_by_pixel_count = success_by_pixel_count.round(2)
    success_by_pixel_count = success_by_pixel_count.applymap(lambda x: f"{x:.2f}%")

    return success_by_pixel_count

def plot_targeted_success_rate_by_pixel_count(success_table):
    """
    Plots a grouped bar chart for targeted attack success rate by model and pixel count.

    Parameters:
    - success_table: DataFrame generated by get_targeted_success_rate_table_by_pixel_count().
                     Rows represent model names, columns represent pixel counts, and
                     values are formatted success rates (e.g., "85.50%").
    """
    # Convert the success_table values back to numeric for plotting
    numeric_table = success_table.replace('%', '', regex=True).astype(float)

    # Plot the grouped bar chart
    numeric_table.plot(kind='bar', stacked=False, figsize=(12, 7), color=palette)

    # Set labels and title
    plt.xlabel('Model Name', fontsize=12)
    plt.ylabel('Attack Success Rate (%)', fontsize=12)
    plt.title('Targeted Attack Success Rate by Model and Pixel Count', fontsize=14)

    # Improve legend and ticks
    plt.legend(title='Pixel Count', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=0)
    plt.tight_layout()

    # Show the plot
    plt.show()



def summary_tables(df, targeted = False):

  """
    Generates summary tables and confusion matrices for attack success rates.

    Parameters:
    - df (DataFrame): Input DataFrame containing the following columns:
    - 'model_name': Name of the model being evaluated.
    - 'pixel_count': Number of perturbed pixels.
    - 'target_class': The intended target class for the attack if targeted is False.
    - 'predicted_class': The class predicted by the model.
    - targeted (bool, optional): Indicates if the attack is targeted (default is False).

    Returns:
    - tuple: A tuple containing:
    1. model_tables (dict): A dictionary where keys are model names, and values are DataFrames.
        Each DataFrame contains pixel counts as columns and success rates as values.
    2. conf_matrix_set (dict): A dictionary where keys are combinations of model names and pixel counts,
        and values are confusion matrices corresponding to those settings.
  """


  # Dictionaries for results
  model_tables = {}
  conf_matrix_set = {}

  # Find model_names and pixel_counts
  model_names = df['model_name'].unique()
  pixel_counts = sorted(df['pixel_count'].unique())

  # Loop through each model and pixel count, calculate success rates and confusion matrices
  for model in model_names:
      model_results = {}

      for pixel in pixel_counts:
          filtered_df = df[(df['model_name'] == model) & (df['pixel_count'] == pixel)]

          # Get success rate by class
          success_rate_class = success_by_class(filtered_df, targeted)

          # Get confusion matrix
          conf_matrix_data = conf_matrix(filtered_df, model, pixel)

          # Store results
          model_results[f"{pixel} Pixel"] = success_rate_class
          conf_matrix_set[f"{model} {pixel} Pixel"] = conf_matrix_data

      # Convert results into a DataFrame and store
      model_tables[model] = pd.DataFrame(model_results)
  return model_tables, conf_matrix_set


def targeted_number_of_target_classes(df):
    """
    Plots a 3x2 grid of histograms for each model, showing the targeted success count per image
    for each pixel count (1 to 6).

    Parameters:
        df (pd.DataFrame): Input DataFrame containing 'model_name', 'pixel_count', 'img_id',
                           and 'targeted_success' columns.

    Returns:
        dict: A dictionary containing success count tables for each model and pixel count combination.
    """
    # Initialize a dictionary to store the success count tables
    tables = {}

    # Get unique model names and pixel counts
    models = df['model_name'].unique()
    pixel_counts = sorted(df['pixel_count'].unique())  # Sort pixel counts for consistent plotting

    for model in models:
        # Create a 3x2 figure layout for the current model
        fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
        fig.suptitle(f"{model}", fontsize=16)
        axes = axes.flatten()

        # Filter data for the current model
        model_data = df[df['model_name'] == model]
        palette = sns.color_palette("hls", 10)

        for idx, pixel in enumerate(pixel_counts):
            ax = axes[idx]

            # Filter the DataFrame for the current pixel count
            pixel_data = model_data[model_data['pixel_count'] == pixel]

            # Get all unique img_ids
            all_img_ids = pixel_data['img_id'].unique()

            # Filter rows where `targeted_success` is True
            successful_attacks = pixel_data[pixel_data['targeted_success'] == True]

            # Count occurrences of `img_id` where `targeted_success` is True
            image_success_counts = (
                successful_attacks.groupby('img_id')
                .size()
                .reindex(all_img_ids, fill_value=0)
                .reset_index(name='success_count')
            )
            tables[f"{model} {pixel}"] = image_success_counts

            # Get the frequency of each success count
            value_counts = image_success_counts['success_count'].value_counts().sort_index()


            # Plot the histogram
            ax.bar(
                image_success_counts['success_count'].value_counts().index,
                image_success_counts['success_count'].value_counts().values,
                color=palette[5],
            )

            # Add the counts above the bars
            for i, count in enumerate(value_counts.values):
                ax.text(
                    value_counts.index[i],  # X-coordinate
                    count + 0.5,  # Y-coordinate (slightly above the bar)
                    str(count),  # Text to display
                    ha='center',  # Horizontal alignment
                    va='bottom',  # Vertical alignment
                    fontsize=9  # Font size
                )

            # Set the title and labels for each subplot
            ax.set_title(f"{pixel} pixel", fontsize=12)
            #ax.set_xlabel("Success Count", fontsize=10)
            ax.set_ylabel("Number of Images", fontsize=10)
            ax.grid(axis="y", linestyle="--", alpha=0.9)
            ax.set_ylim([0, 100])
            ax.grid(False)
            for ax in axes.flat:
              ax.label_outer()

        # Hide any extra subplots
        for idx in range(len(pixel_counts), len(axes)):
            fig.delaxes(axes[idx])

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()


def broad_targeting_success_rate(df):
    """
    Computes the average number of images successfully attacked at least once
    and provides the results as a pivot table.

    Parameters:
        df (pd.DataFrame): The input DataFrame with columns including `model_name`,
                           `pixel_count`, `img_id`, and `targeted_success`.

    Returns:
        pd.DataFrame: A pivot table in the format:
                      model_name | PixelCount1 | PixelCount2 | ...
    """
    # Filter rows where `targeted_success` is True
    success_df = df[df['targeted_success'] == True]

    # Group by `model_name`, `pixel_count`, and count unique `img_id`
    counts = success_df.groupby(['model_name', 'pixel_count'])['img_id'].nunique().reset_index(name='successful_images')

    # Compute the average successful attacks grouped by `model_name` and `pixel_count`
    average_counts = counts.groupby(['model_name', 'pixel_count'])['successful_images'].mean().reset_index()
    average_counts.rename(columns={'successful_images': 'avg_successful_attacks'}, inplace=True)
    # Convert `avg_successful_attacks` to percentages
    average_counts['avg_successful_attacks'] = average_counts['avg_successful_attacks'].round(2).astype(str) + '%'

    # Create the pivot table
    pivot_table = average_counts.pivot(index='model_name', columns='pixel_count', values='avg_successful_attacks')

    return pivot_table
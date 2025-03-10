# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# class DataAugmenter:
#     """
#     A class for augmenting and manipulating trajectory data stored in CSV files.
#     This class provides functionality to augment trajectory data through various transformations,
#     including translations, flips, and combinations of both. It's designed to work with CSV files
#     containing coordinate data with '_x' and '_y' suffixes for position columns.
#     Attributes:
#         input_csv (str): Path to the input CSV file containing original trajectory data.
#         output_csv (str): Path where the augmented data will be saved (optional).
#         transformations (list): List of tuples containing (dx, dy) transformation pairs.
#         offsets (list): List of scaling factors applied to transformations.
#         df (pandas.DataFrame): Original data loaded from input CSV.
#         augmented_df (pandas.DataFrame): Data after applying transformations.
#         df_flipped (pandas.DataFrame): Flipped version of the augmented data.
#         df_combined (pandas.DataFrame): Combined original and flipped data.
#     Methods:
#         augment_normal(): Applies translations to create augmented dataset.
#         flip_data(): Creates a mirrored version of the augmented data.
#         combine_data(): Merges original augmented and flipped datasets.
#         save_to_csv(file_path=None): Saves the processed data to a CSV file.
#         plot_trajectories(output_file=None, sample_size=1000): Visualizes the trajectories.
#     Example:
#         >>> augmenter = DataAugmenter('input.csv', 'output.csv')
#         >>> augmented_data = augmenter.augment_normal()
#         >>> flipped_data = augmenter.flip_data()
#         >>> combined_data = augmenter.combine_data()
#         >>> augmenter.save_to_csv()
#     Raises:
#         FileNotFoundError: If the input CSV file doesn't exist.
#         ValueError: If the input CSV is empty or if attempting to save/plot without processing data first.
#     """

#     def __init__(self, input_csv, output_csv=None, transformations=None, offsets=None):
#         """
#         Initialize the DataAugmenter with input and output CSV paths, transformations, and offsets.
        
#         :param input_csv: Path to the input CSV file
#         :param output_csv: Path to save the augmented CSV file (optional)
#         :param transformations: List of transformation tuples (default is predefined set)
#         :param offsets: List of offset values (default is [15, 5])
#         """
#         self.input_csv = input_csv
#         self.output_csv = output_csv

#         self.transformations = transformations if transformations else [
#             (0, 0), (1, 1), (1, -1), (-1, 1), (-1, -1),
#             (2, 0), (-2, 0), (0, 2), (0, -2),
#             (2, 2), (2, -2), (-2, 2), (-2, -2),
#             (3, 1), (3, -1), (-3, 1), (-3, -1),
#             (1, 3), (1, -3), (-1, 3), (-1, -3),
#             (3, 3), (3, -3), (-3, 3), (-3, -3), 
#             (3,-5), (-3,-5), (3,5), (-3,5), (-5,3), (-5,-3), (5,3), (5,-3),
#             (4, 0), (-4, 0), (0, 4), (0, -4),
#             (4, 2), (4, -2), (-4, 2), (-4, -2),
#             (2, 4), (2, -4), (-2, 4), (-2, -4),
#             (4, 4), (4, -4), (-4, 4), (-4, -4),
#             (5, 1), (5, -1), (-5, 1), (-5, -1),
#             (1, 5), (1, -5), (-1, 5), (-1, -5)
#         ]

#         self.offsets = offsets if offsets else [15, 5]

#         try:
#             self.df = pd.read_csv(self.input_csv)
#             if self.df.empty:
#                 raise ValueError(f"Input file {self.input_csv} is empty.")
#         except FileNotFoundError:
#             raise FileNotFoundError(f"Input file {self.input_csv} not found.")
#         except pd.errors.EmptyDataError:
#             raise ValueError(f"Input file {self.input_csv} is empty.")

#         self.augmented_df = pd.DataFrame()
#         self.df_flipped = None
#         self.df_combined = None

#     def augment_normal(self):
#         """
#         Augments the data by applying transformations and offsets in a vectorized manner.
#         """
#         df_expanded = self.df.copy()
#         for dx, dy in self.transformations:
#             for offset in self.offsets:
#                 df_transformed = self.df.copy()
#                 df_transformed.filter(like='_x') += dx * offset
#                 df_transformed.filter(like='_y') += dy * offset
#                 df_expanded = pd.concat([df_expanded, df_transformed], ignore_index=True)
        
#         self.augmented_df = df_expanded
#         return self.augmented_df

#     def flip_data(self):
#         """
#         Flips the augmented data along the y-axis using vectorized operations.
#         """
#         if self.augmented_df.empty:
#             self.augment_normal()

#         x_columns = [col for col in self.augmented_df.columns if '_x' in col]
#         center_x = self.augmented_df[x_columns].mean()

#         self.df_flipped = self.augmented_df.copy()
#         self.df_flipped[x_columns] = 2 * center_x - self.df_flipped[x_columns]

#         c_columns = [col for col in self.augmented_df.columns if col.endswith('_c')]
#         self.df_flipped[c_columns] = -self.df_flipped[c_columns]

#         return self.df_flipped

#     def combine_data(self):
#         """
#         Combines the original augmented data with the flipped data.
#         """
#         if self.df_flipped is None:
#             self.flip_data()
        
#         self.df_combined = pd.concat([self.augmented_df, self.df_flipped], ignore_index=True)
#         return self.df_combined

#     def save_to_csv(self, file_path=None):
#         """
#         Saves the augmented or combined data to a CSV file.
        
#         :param file_path: Path to save the CSV (uses self.output_csv if not provided)
#         """
#         if self.df_combined is not None:
#             save_df = self.df_combined
#         elif self.augmented_df is not None and not self.augmented_df.empty:
#             save_df = self.augmented_df
#         else:
#             raise ValueError("No data to save. Run augment_data() first.")

#         save_path = file_path or self.output_csv
#         if save_path is None:
#             raise ValueError("No output file path specified")

#         save_df.to_csv(save_path, index=False)
#         print(f"Data saved to: {save_path}")
#         return save_path

#     def plot_trajectories(self, output_file=None, sample_size=1000):
#         """
#         Plot the original and flipped trajectories.
        
#         :param output_file: Optional path to save the plot
#         """
#         if self.augmented_df is None or self.df_flipped is None:
#             raise ValueError("Run augment_data() and flip_data() before plotting")

#         x_columns = [col for col in self.augmented_df.columns if '_x' in col]
#         y_columns = [col for col in self.augmented_df.columns if '_y' in col]

#         plt.figure(figsize=(12, 8))

#         sampled_df = self.augmented_df.sample(min(sample_size, len(self.augmented_df)))
#         sampled_flipped = self.df_flipped.sample(min(sample_size, len(self.df_flipped)))

#         for x_col, y_col in zip(x_columns, y_columns):
#             plt.scatter(sampled_df[x_col], sampled_df[y_col], label=f"Original {x_col.replace('_x', '')}", alpha=0.7)
#             plt.scatter(sampled_flipped[x_col], sampled_flipped[y_col], label=f"Flipped {x_col.replace('_x', '')}", alpha=0.5)

#         plt.xlabel("X Coordinates")
#         plt.ylabel("Y Coordinates")
#         plt.title("Scatter Plot of X vs Y Coordinates (Original and Flipped)")
#         plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1))
#         plt.grid(True)
#         plt.tight_layout()

#         if output_file:
           


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class DataAugmenter:
    def __init__(self, input_csv, output_csv=None, transformations=None, offsets=None):
        """
        Initialize the DataAugmenter with input and output CSV paths, transformations, and offsets.
        
        :param input_csv: Path to the input CSV file
        :param output_csv: Path to save the augmented CSV file (optional)
        :param transformations: List of transformation tuples (default is predefined set)
        :param offsets: List of offset values (default is [15, 5])
        """
        self.input_csv = input_csv
        self.output_csv = output_csv
        
        # Default transformations if not provided
        self.transformations = transformations if transformations else [
            (0, 0), 
            (1, 1), (1, -1), (-1, 1), (-1, -1),
            (2, 0), (-2, 0), (0, 2), (0, -2),
            (2, 2), (2, -2), (-2, 2), (-2, -2),
            (3, 1), (3, -1), (-3, 1), (-3, -1),
            (1, 3), (1, -3), (-1, 3), (-1, -3),
            (3, 3), (3, -3), (-3, 3), (-3, -3), 
            (3,-5), (-3,-5),(3,5),(-3,5),(-5,3),(-5,-3),(5,3),(5,-3),
            (4, 0), (-4, 0), (0, 4), (0, -4),
            (4, 2), (4, -2), (-4, 2), (-4, -2),
            (2, 4), (2, -4), (-2, 4), (-2, -4),
            (4, 4), (4, -4), (-4, 4), (-4, -4),
            (5, 1), (5, -1), (-5, 1), (-5, -1),
            (1, 5), (1, -5), (-1, 5), (-1, -5)
        ]
        
        # Default offsets if not provided
        self.offsets = offsets if offsets else [15, 5]
        
        # Load the input data
        self.df = pd.read_csv(self.input_csv)
        
        # Initialize augmented DataFrame
        self.augmented_df = pd.DataFrame()
        
        # Flipped DataFrame
        self.df_flipped = None
        self.df_combined = None

    def augment_normal(self):
        """
        Augments the data by applying transformations and offsets to each row in the CSV.
        """
        for index, row in self.df.iterrows():
            # Add original row
            self.augmented_df = pd.concat([self.augmented_df, row.to_frame().T], ignore_index=True)
            
            # Apply transformations and offsets
            for dx, dy in self.transformations:
                for offset in self.offsets:
                    # Create a copy of the row
                    new_row = row.copy()
                    
                    # Modify x and y coordinates
                    for col in self.df.columns:
                        if '_x' in col:
                            new_row[col] += (dx * offset)
                        if '_y' in col:
                            new_row[col] += (dy * offset)
                    
                    # Add transformed row
                    self.augmented_df = pd.concat([self.augmented_df, new_row.to_frame().T], ignore_index=True)
        
        return self.augmented_df

    def flip_data(self):
        """
        Flips the augmented data along the y-axis.
        """
        # If augmentation hasn't been done, do it first
        if self.augmented_df.empty:
            self.augment_normal()
        
        # Extract columns for flipping
        x_columns = [col for col in self.augmented_df.columns if '_x' in col]
        y_columns = [col for col in self.augmented_df.columns if '_y' in col]
        
        # Calculate center point
        center_x = np.mean(self.augmented_df[x_columns].values)
        center_y = np.mean(self.augmented_df[y_columns].values)
        
        # Create a copy for flipped data
        self.df_flipped = self.augmented_df.copy()
        
        # Flip x-coordinates around center_x
        for x_col, y_col in zip(x_columns, y_columns):
            self.df_flipped[x_col] = 2 * center_x - self.augmented_df[x_col]
            # y-coordinates remain unchanged
            self.df_flipped[y_col] = self.augmented_df[y_col]
        
        # Flip the sign of columns ending with '_c'
        c_columns = [col for col in self.augmented_df.columns if col.endswith('_c')]
        for col in c_columns:
            self.df_flipped[col] = -self.augmented_df[col]
        
        return self.df_flipped

    def combine_data(self):
        """
        Combines the original augmented data with the flipped data.
        
        :return: Combined DataFrame
        """
        # If flipping hasn't been done, do it first
        if self.df_flipped is None:
            self.flip_data()
        
        # Combine original augmented and flipped data
        self.df_combined = pd.concat([self.augmented_df, self.df_flipped], ignore_index=True)
        
        return self.df_combined

    def save_to_csv(self, file_path=None):
        """
        Saves the augmented or combined data to a CSV file.
        
        :param file_path: Path to save the CSV (uses self.output_csv if not provided)
        """
        # Determine which DataFrame to save
        if self.df_combined is not None:
            save_df = self.df_combined
        elif self.augmented_df is not None and not self.augmented_df.empty:
            save_df = self.augmented_df
        else:
            raise ValueError("No data to save. Run augment_data() first.")
        
        # Use provided file path or default output path
        save_path = file_path or self.output_csv
        
        if save_path is None:
            raise ValueError("No output file path specified")
        
        save_df.to_csv(save_path, index=False)
        print(f"Data saved to: {save_path}")
        return save_path

    def plot_trajectories(self, output_file=None):
        """
        Plot the original and flipped trajectories.
        
        :param output_file: Optional path to save the plot
        """
        # Ensure we have data to plot
        if self.augmented_df is None or self.df_flipped is None:
            raise ValueError("Run augment_data() and flip_data() before plotting")
        
        # Extract x and y columns
        x_columns = [col for col in self.augmented_df.columns if '_x' in col]
        y_columns = [col for col in self.augmented_df.columns if '_y' in col]
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        
        # Plot original and flipped trajectories
        for x_col, y_col in zip(x_columns, y_columns):
            plt.scatter(self.augmented_df[x_col], self.augmented_df[y_col], 
                        label=f"Original {x_col.replace('_x', '')}", alpha=0.7)
            plt.scatter(self.df_flipped[x_col], self.df_flipped[y_col], 
                        label=f"Flipped {x_col.replace('_x', '')}", alpha=0.5)
        
        plt.xlabel("X Coordinates")
        plt.ylabel("Y Coordinates")
        plt.title("Scatter Plot of X vs Y Coordinates (Original and Flipped along Y-axis)")
        plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1))
        plt.grid(True)
        plt.tight_layout()
        
        # Save or show the plot
        if output_file:
            plt.savefig(output_file)
            plt.close()
        else:
            plt.show()

    def get_augmented_data_samples(self):
        """
        Returns the number of rows in the augmented DataFrame.
        
        :return: Number of rows in augmented data
        """
        return len(self.augmented_df)

    def extract_nth_samples(self, n, new_output_csv):
        """
        Extracts every nth sample from the augmented data and saves it to a new CSV.
        
        :param n: Extract every nth sample
        :param new_output_csv: Path to save the sampled CSV
        :return: Path to the saved CSV
        """
        if self.augmented_df.empty:
            raise ValueError("No augmented data. Run augment_data() first.")
        
        total_samples = len(self.augmented_df)
        step = max(1, total_samples // n)
        sampled_df = self.augmented_df.iloc[::step]
        sampled_df.to_csv(new_output_csv, index=False)
        return new_output_csv
    
    def augment_data(self):
        self.augment_normal()
        self.flip_data()
        self.combine_data()
        return self.df_combined
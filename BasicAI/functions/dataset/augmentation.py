"""
This codes takes in a CSV and augments the data by applying transformations and offsets to each row in the CSV.
"""
import pandas as pd

class DataAugmenter:
    def __init__(self, input_csv, output_csv, transformations=None, offsets=None):
        self.input_csv = input_csv
        self.output_csv = output_csv
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
        self.offsets = offsets if offsets else [17]
        self.df = pd.read_csv(self.input_csv)
        self.augmented_df = pd.DataFrame()

    def augment_data(self):
        """
        Augments the data by applying transformations and offsets to each row in the CSV.
        """
        for index, row in self.df.iterrows():
            self.augmented_df = pd.concat([self.augmented_df, row.to_frame().T], ignore_index=True)
            for dx, dy in self.transformations:
                for offset in self.offsets:
                    new_row = row.copy()
                    for col in self.df.columns:
                        if '_x' in col:
                            new_row[col] += (dx * offset) 
                        if '_y' in col:
                            new_row[col] += (dy * offset)
                    self.augmented_df = pd.concat([self.augmented_df, new_row.to_frame().T], ignore_index=True)

    def save_to_csv(self):
        """
        Saves the augmented data to a CSV.
        """
        self.augmented_df.to_csv(self.output_csv, index=False)
    
    def get_augmented_data_samples(self):
        """
        Returns the number of rows in the augmented DataFrame.
        """
        return len(self.augmented_df)

    def get_augmented_object_locations(self):
        """
        """
        object_1_locations = []
        object_2_locations = []
        
        for index, row in self.augmented_df.iterrows():
            object_1_locations.append([row[0], row[1], row[2]])
            object_2_locations.append([row[3], row[4], row[5]])
        
        return object_1_locations, object_2_locations

    def extract_nth_samples(self, n, new_output_csv):
        """
        Extracts every nth sample from the augmented data and saves it to a new CSV.
        """
        total_samples = len(self.augmented_df)
        step = max(1, total_samples // n)
        sampled_df = self.augmented_df.iloc[::step]
        sampled_df.to_csv(new_output_csv, index=False)
        return new_output_csv
# Usage Example:
# augmenter = DataAugmenter(input_csv='input.csv', output_csv='augmented_output.csv')
# augmenter.augment_data()
# augmenter.save_to_csv()

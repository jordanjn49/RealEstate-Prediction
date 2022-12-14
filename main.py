import data_graphic
import data_process

if __name__ == "__main__":

    # We read and filter columns from dataset file
    # dataset = data_process.preprocessing('DATASET-VF2022S1.txt')

    # After going to https://adresse.data.gouv.fr/csv, we get the geocoded CSV, and now we are going to filter it!
    dataset_geo = data_process.add_coordinates('DATASET-Preprocessed&Geocoded.csv')

    dataset_neighborhood = data_process.determine_neighborhood(dataset_geo)

    dataset_graphic = data_graphic.plotAll('DATASET-Preprocessed&Geocoded&Filtered.csv')
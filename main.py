import data_graphic
import data_process

if __name__ == "__main__":
    # We read and filter columns from dataset file
    # data_process.preprocessing('DATASET-VF2022S1.txt')

    # After going to https://adresse.data.gouv.fr/csv, we get the geocoded CSV, and now we are going to filter it!
    data_process.feature_engineering('DATASET-Preprocessed&Geocoded.csv')

    data_graphic.plotAll('DATASET-Final.csv')

import data_process

if __name__ == "__main__":

    # We read and filter columns from dataset file
    dataset = data_process.preprocessing('DATASET-VF2022S1.txt')

    # Adding coordinates with Geo-Coding
    dataset_geo = data_process.add_coordinates(dataset)

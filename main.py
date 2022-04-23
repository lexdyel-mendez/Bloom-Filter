import numpy as np
import pandas as pd
import csv, mmh3, math, sys
from tqdm import tqdm


# This is a program designed to generate a Bloom Filter respective to a given Data Base, and then it will verify
# which of the values exist, from another database. A Bloom filter is a space-efficient probabilistic data structure
# that is used to test whether an element is a member of a set. False positive matches are possible,
# but false negatives are not – in other words, a query returns either "possibly in set" or "definitely not in set".


# Author: Lexdyel J. Mendez Rios
# TimeStamp: 3/18/2022
# Course: CIIC 4025
# Prof. Wilfredo Lugo

class BloomFilter():
    """
    The function takes in an array of data and a false probability, and returns a Bloom Filter object with the filter
    size, hash count, and filter array.

    :param input_data: The data that we want to store in the Bloom Filter
    :type input_data: np.ndarray
    :param falseProb: The probability of false positives
    :type falseProb: float
    """

    def __init__(self, input_data: np.ndarray, falseProb: float):
        self.input_data = input_data
        self.items_count = len(input_data)
        self.falseProb = falseProb
        self.filterSize = self.__getSize(self.items_count,
                                         falseProb)

        self.hashCount = self.__getHashCount(self.filterSize,
                                             self.items_count)

        self.filterArray = np.zeros(self.filterSize)

    def __getSize(self, n: int, p: float) -> int:
        """
        The function takes in the number of items and the probability of false positives and returns the size of the bit
        array that will be used as the Bloom Filter. This comes from the mathematical formula:

        m = ceil((n * log(p)) / log(1 / pow(2, log(2))))

        :param n: The number of items in the filter
        :type n: int
        :param p: the probability of false positives, a decimal value between 0 and 1. (1% = 0.01, 5% = 0.05, etc.)
        :type p: float
        :return: The size of the bloom filter
        """

        m = (n * math.log(p)) / math.log(1 / math.pow(2, math.log(2)))
        return math.ceil(m)

    def __getHashCount(self, m: int, n: int) -> int:
        """
        The function takes in two integers, m and n, and returns the number of hash functions needed to create a bloom
        filter with m bits and n elements. The mathematical formula for this is:

        k = round((m / n) * log(2));

        :param m: Size of the bit array
        :type m: int
        :param n: number of items in the set
        :type n: int
        :return: The number of hash functions to be used.
        """
        k = (m / n) * math.log(2)
        return int(k)

    def addItem(self, item):
        """
        We are taking the item, hashing it with the hash function, and then setting the bit at the index of the hash
        to 1. The amount of hashes that it makes depend on how accurate one wants the filter to be.

        :param item: the item to be added to the filter
        """

        # Creating an empty list that will be used to store the hashes of the items.
        hashes = []
        # Loop that will run as many times as the number of hash functions that we want to use.
        for i in range(self.hashCount):
            # Taking the item and hashing it with the hash function.
            hashing = mmh3.hash(item[0], i) % self.filterSize
            # Adding the hash value to the list of hashes.
            hashes.append(hashing)
            # Setting the bit at the index of the hash to 1.
            self.filterArray[hashing] = 1

    def verify(self, item):
        """
        Will verify if the given value exist in the original database. If any of the hash values of the item are in
        the filter, return True. Otherwise, return False.

        :param item: the item to be verified
        :return: True or False
        """

        for i in range(self.hashCount):
            # Taking the item and hashing it with the hash function.
            hashing = mmh3.hash(item[0], i) % self.filterSize
            # Checking if the bit at the index of the hash is 0. If it is, then it returns False.
            if self.filterArray[hashing] == 0: return False

        # Returning True if the item is in the filter.
        return True


def data_extraction(filepath: str) -> np.ndarray:
    """
    This function takes a filepath and returns the data in the file as a numpy array

    :param filepath: the path to the file you want to read in
    :type filepath: str
    :return: a numpy array of the data.
    """
    data = pd.read_csv(filepath)
    return data.to_numpy()


def feeder(db_input: np.ndarray, db_check: np.ndarray, new_filename='results.csv', acc=1e-7):
    """
    It takes in two arrays, one for the database and one for the check, and then it creates a bloom filter with the
    database array, and then it checks each item in the check array to see if it is in the database. The results for
    each query are saved in the results.csv file.

    :param db_input: The database of emails that we want to add to the filter
    :type db_input: np.ndarray
    :param db_check: The list of emails to check against the database
    :type db_check: np.ndarray
    :param new_filename: The name of the file that will be created, defaults to results.csv (optional)
    :param acc: The accuracy of the bloom filter
    """
    header = ["Email", "Result"]

    bloomFilter = BloomFilter(db_input, acc)

    # Adding each item in the database to the bloom filter.
    print("Adding Data to Bloom Filter")
    for item in tqdm(db_input):
        bloomFilter.addItem(item)


    with open(new_filename, 'w', newline='', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(header)  # Add headers to the file
        # Checking each email in the db_check array to see if it is in the database.
        for email in tqdm(db_check):
            if bloomFilter.verify(email):
                # If any bit is 1, then there is a probability that it exits
                writer.writerow([email[0], "Probably in the DB"])
                # Otherwise, it is not present in the filter.
            else:
                writer.writerow([email[0], "Not  in the DB"])
        f.close()


def main():
    """
    It takes in two filepaths, one for the input file and one for the check file, and returns two dataframes,
    one for the input file and one for the check file
    """
    # Taking the first and second argument from the command line and assigning them to the variables input_filepath and
    # check_filepath.
    input_filepath = sys.argv[1]
    check_filepath = sys.argv[2]

    # Taking the first and second argument from the command line and assigning them to the variables input_filepath and
    # check_filepath.
    db_input = data_extraction(input_filepath)
    db_check = data_extraction(check_filepath)

    # Taking the two arrays, one for the database and one for the check, and then it creates a bloom filter with the
    # database array, and then it checks each item in the check array to see if it is in the database. The results for
    # each query are saved in the results.csv file.
    feeder(db_input, db_check)


if __name__ == '__main__':
    main()

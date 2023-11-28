import argparse

from utils import DataFileParser

parser = argparse.ArgumentParser()
parser.add_argument(
    "-t",
    "--training",
    default="data_files/training",
    help="The path to the trainning file",
)
parser.add_argument(
    "-i",
    "--input",
    default="data_files/testing",
    help="The path to the test prediction file",
)
parser.add_argument(
    "-l",
    "--likelihood",
    default="data_files/likelihood",
    help="The path to the likelihood file",
)

if __name__ == "__main__":
    args = parser.parse_args()
    training_file_path = getattr(args, "training")
    input_file_path = getattr(args, "input")
    likelihood_file_path = getattr(args, "likelihood")

    likelihood = DataFileParser.parse_likelihood_file(likelihood_file_path)

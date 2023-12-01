import argparse

from models import NaiveBayesClassifier
from utils import DataFileParser
from utils.constants import INITIAL_PROBS, TRANSITION_PROBS

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
parser.add_argument(
    "-eaf",
    "--enableAdditionalFeature",
    action="store_true",
    help="Flag to enable the additional feature",
)

if __name__ == "__main__":
    args = parser.parse_args()
    training_file_path = getattr(args, "training")
    input_file_path = getattr(args, "input")
    likelihood_file_path = getattr(args, "likelihood")
    enable_additional_feature = getattr(args, "enableAdditionalFeature")

    # Parse file input
    likelihood = DataFileParser.parse_likelihood_file(likelihood_file_path)
    inputs = DataFileParser.parse_input_file(input_file_path)
    training = DataFileParser.parse_trainning_file(training_file_path)

    classifier = NaiveBayesClassifier(
        INITIAL_PROBS, TRANSITION_PROBS, enable_additional_feature
    )
    if not enable_additional_feature:
        predictions = []
        classifier.fit(likelihood, ["Bird", "Plane"])
        predictions = classifier.predict(training[:10])

        for i, result in enumerate(predictions):
            print(f"Object {i + 1}: {result}")

        predictions = classifier.predict(training[10:])

        for i, result in enumerate(predictions):
            print(f"Object {i + 1}: {result}")
    else:
        classifier.fit(training, ["Bird"] * 10 + ["Plane"] * 10)
        predictions = classifier.predict(inputs)
        for i, result in enumerate(predictions):
            print(f"Object {i + 1}: {result}")

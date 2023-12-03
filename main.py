import argparse

from models import NaiveBayesClassifier
from utils import DataFileParser, pretty_print, labeling_training_data_for_extra_feature
from utils.constants import TRANSITION_PROBS

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
parser.add_argument(
    "-pat",
    "--predictAgainstTraining",
    action="store_true",
    help="Flag to tell the classifier whether to use training file for testing",
)

if __name__ == "__main__":
    args = parser.parse_args()
    training_file_path = getattr(args, "training")
    input_file_path = getattr(args, "input")
    likelihood_file_path = getattr(args, "likelihood")
    enable_additional_feature = getattr(args, "enableAdditionalFeature")
    predict_against_training = getattr(args, "predictAgainstTraining")

    # Parse file input
    likelihood = DataFileParser.parse_likelihood_file(likelihood_file_path)
    inputs = DataFileParser.parse_input_file(input_file_path)
    training = DataFileParser.parse_trainning_file(training_file_path)

    classifier = NaiveBayesClassifier(TRANSITION_PROBS, enable_additional_feature)
    if not enable_additional_feature:
        predictions = []
        classifier.train(likelihood, ["Bird", "Plane"])
    else:
        labeling = labeling_training_data_for_extra_feature(training, 0.5)
        classifier.train(training, labeling)

    if predict_against_training:
        predictions = classifier.predict(training)
    else:
        predictions = classifier.predict(inputs)

    pretty_print(predictions)

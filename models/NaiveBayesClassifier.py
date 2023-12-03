import math
from statistics import mean, variance
from collections import defaultdict

from utils.constants import INITIAL_PROBS


class NaiveBayesClassifier:
    """
    The Class representing the Naive Bayes Classifier
    Properties:
        __likelihood: The likelihood that was given before hand
        __mean_variance_prior_by_class: The mean, variance and initial probability of each prior feature
        __trainsition_probs: The transition probability
        __classes: The list of classes that was seen in the given label when training
        __enable_addition_feature: The flag to indicate whether the algorithm should predict with the training data provided
    """
    def __init__(
        self,
        transition_probs: float,
        enable_addition_feature: bool,
    ):
        self.__likelihood = dict()
        self.__mean_variance_prior_by_class: dict[str, list[float]] = dict()
        self.__transition_probs: float = transition_probs
        self.__classes: list[str] = []
        self.__enable_addition_feature = enable_addition_feature

    def train(self, priors: list[list[float]], expected_outcome: list[str]):
        """
        The function to train the model of the existing knowledge.
        Arguments:
            priors: the list of priors collected data.
            expected_outcome: the list of labeling that was categorized before.
        """
        if not self.__enable_addition_feature:
            self.__train_with_given_likelihood(expected_outcome, priors)
        else:
            self.__train_with_prior_data(expected_outcome, priors)

        return self

    def predict(self, measurements: list[list[float]]):
        """
        The function to predict the given list of records.
        Arguments:
            measurements: The list of records to predict.
        Returns:
            The prediction of all the records.
        """
        predictions = []
        for measurement in measurements:
            predictions.append(self.__predict_feature(measurement))

        return predictions

    def __likelihood_func(self, clazz: str, measurement: float) -> float:
        """
        The Likelihood function which determines whether to returns the value from the given likelihood list or
        calculate the likelihood using Guassian Distribution formula.

        Arguments:
            clazz: the current clazz that we're calculating the probability on.
            measurement: the current feature to calculate the likelihood for.

        Returns:
            The likelihood value of the current feature with the given class.
        """
        if not self.__enable_addition_feature:
            return self.__likelihood[clazz][
                self.__calculate_likelihood_index(measurement)
            ]

        mean, variance, _ = self.__mean_variance_prior_by_class[clazz]
        return (
            1
            / (math.sqrt(2 * math.pi * variance))
            * math.exp(-((measurement - mean) ** 2 / (2 * variance)))
        )


    def __train_with_given_likelihood(
        self, expected_outcome: list[str], priors: list[list[float]]
    ):
        for i, clazz in enumerate(expected_outcome):
            if clazz not in self.__classes:
                self.__classes.append(clazz)

            self.__likelihood[clazz] = priors[i]

    def __train_with_prior_data(
        self, expected_outcome: list[str], priors: list[list[float]]
    ):
        (
            classes_historical_records,
            classes_occurence_count,
        ) = self.__group_by_class_and_count_occurence(expected_outcome, priors)

        for clazz in classes_historical_records:
            if clazz not in self.__mean_variance_prior_by_class:
                self.__mean_variance_prior_by_class[clazz] = list()

            prior_probability = classes_occurence_count[clazz] / len(expected_outcome)
            self.__mean_variance_prior_by_class[clazz] = [
                mean(classes_historical_records[clazz]),
                variance(classes_historical_records[clazz]),
                prior_probability,
            ]

    def __predict_feature(self, measurement: list[float]):
        historical_probs = defaultdict(list)

        initial_probabilities = self.__calculate_initial_probability(measurement[0])

        for clazz in initial_probabilities:
            historical_probs[clazz].append(initial_probabilities[clazz])

        for t in range(1, len(measurement)):
            current_probability = self.__calculate_current_probability(
                measurement[t], t, historical_probs
            )
            for clazz in historical_probs:
                historical_probs[clazz].append(current_probability[clazz])

        return self.__get_prediction(historical_probs)

    def __group_by_class_and_count_occurence(
        self, expected_outcome: list[str], priors: list[list[float]]
    ):
        classes_historical_records = defaultdict(list)
        classes_occurence_count = defaultdict(int)
        for i, clazz in enumerate(expected_outcome):
            if clazz not in self.__classes:
                self.__classes.append(clazz)

            classes_historical_records[clazz] = (
                classes_historical_records[clazz] + priors[i]
            )
            classes_occurence_count[clazz] += 1

        return classes_historical_records, classes_occurence_count

    def __get_inital_probs(self, clazz: str):
        if not self.__enable_addition_feature:
            return INITIAL_PROBS

        return self.__mean_variance_prior_by_class[clazz][
            len(self.__mean_variance_prior_by_class[clazz]) - 1
        ]

    def __calculate_current_probability(
        self, measurement: float, t: int, historical_probs: dict[str, list[float]]
    ):
        Bts = defaultdict(float)
        for clazz in self.__classes:
            Bts[clazz] = self.__likelihood_func(clazz, measurement) * (
                (historical_probs[clazz][t - 1] * self.__transition_probs)
                + (historical_probs[clazz][t - 1] * (1 - self.__transition_probs))
            )

        return self.__normalize_B(Bts)

    def __get_prediction(self, recorded_prob: dict[str, list[float]]):
        max_prob = 0
        prediction = self.__classes[0]
        for clazz in recorded_prob:
            if recorded_prob[clazz][len(recorded_prob[clazz]) - 1] > max_prob:
                max_prob = recorded_prob[clazz][len(recorded_prob[clazz]) - 1]
                prediction = clazz

        return prediction

    def __calculate_initial_probability(self, measurement: float) -> dict[str, float]:
        B0s = defaultdict(float)
        for clazz in self.__classes:
            B0s[clazz] = self.__likelihood_func(
                clazz, measurement
            ) * self.__get_inital_probs(clazz)

        return self.__normalize_B(B0s)

    def __normalize_B(self, Bs: dict[str, float]) -> dict[str, float]:
        res = defaultdict(float)
        sum_B0s = sum(Bs.values())
        for clazz in Bs:
            res[clazz] = Bs[clazz] / sum_B0s

        return res

    def __calculate_likelihood_index(self, measurement: float) -> int:
        return (round(measurement) * 2) - 1

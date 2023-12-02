import math
from statistics import mean, variance
from collections import defaultdict


class NaiveBayesClassifier:
    def __init__(
        self,
        initial_probs: float,
        transition_probs: float,
        enable_addition_feature: bool,
    ):
        self.__likelihood: dict[str, list[list[float]]] = dict()
        self.__initial_probs: float = initial_probs
        self.__transition_probs: float = transition_probs
        self.__classes: list[str] = []
        self.__enable_addition_feature = enable_addition_feature

    def fit(self, priors: list[list[float]], expected_outcome: list[str]):
        if not self.__enable_addition_feature:
            for i, clazz in enumerate(expected_outcome):
                if clazz not in self.__classes:
                    self.__classes.append(clazz)

                self.__likelihood[clazz].append(priors[i])
        else:
            classes_historical_records = defaultdict(list)
            for i, clazz in enumerate(expected_outcome):
                if clazz not in self.__classes:
                    self.__classes.append(clazz)

                classes_historical_records[clazz].append(priors[i])

            for clazz in classes_historical_records:
                class_historical_records: list[
                    list[float]
                ] = classes_historical_records[clazz]
                for record in class_historical_records:
                    if clazz not in self.__likelihood:
                        self.__likelihood[clazz] = list()

                    self.__likelihood[clazz].append(
                        [
                            mean(record),
                            variance(record),
                        ]
                    )
        return self

    def predict(self, measurements: list[list[float]]):
        predictions = []
        for measurement in measurements:
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

            predictions.append(self.__get_prediction(historical_probs))

        return predictions

    def __likelihood_func(self, clazz: str, measurement: float) -> float:
        if not self.__enable_addition_feature:
            return self.__likelihood[clazz][0][
                self.__calculate_likelihood_index(measurement)
            ]

        features_means_variances = self.__likelihood[clazz]
        likelihood = 0
        for mean, variance in features_means_variances:
            new_likelihood = (
                1
                / (math.sqrt(2 * math.pi * variance))
                * math.exp(-((measurement - mean) ** 2 / (2 * variance)))
            )

            if new_likelihood > likelihood:
                likelihood = new_likelihood

        return likelihood

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
            B0s[clazz] = (
                self.__likelihood_func(clazz, measurement) * self.__initial_probs
            )

        return self.__normalize_B(B0s)

    def __normalize_B(self, Bs: dict[str, float]) -> dict[str, float]:
        res = defaultdict(float)
        sum_B0s = sum(Bs.values())
        for clazz in Bs:
            res[clazz] = Bs[clazz] / sum_B0s

        return res

    def __calculate_likelihood_index(self, measurement: float) -> int:
        return (round(measurement) * 2) - 1

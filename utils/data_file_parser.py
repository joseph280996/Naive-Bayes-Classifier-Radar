import math
from collections import defaultdict


class DataFileParser:
    @staticmethod
    def parse_likelihood_file(path: str) -> list[list[float]]:
        with open(f"{path}.txt", "r") as file:
            lines = file.readlines()
            res = []
            for line in lines:
                res.append(list(map(float, line.strip().split(" "))))

            return res

    @staticmethod
    def parse_input_file(path: str) -> list[list[float]]:
        with open(f"{path}.txt", "r") as file:
            lines = file.readlines()
            res = []
            for line in lines:
                speed_list = list(map(float, line.strip().split(" ")))
                mean = DataFileParser.__mean(speed_list)
                removed_nan_list = list(
                    map(lambda x: mean if math.isnan(x) else x, speed_list)
                )
                res.append(removed_nan_list)

            return res

    @staticmethod
    def parse_trainning_file(path: str, distribution_rate: float = 0.5) -> list[list[float]]:
        training_content = DataFileParser.parse_input_file(path)
        return training_content

    @staticmethod
    def __mean(nums: list[float]) -> float:
        sum = 0
        count = 0
        for num in nums:
            if not math.isnan(num):
                sum += num
                count += 1
        return sum / count

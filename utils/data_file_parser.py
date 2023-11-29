class DataFileParser:

    @staticmethod
    def parse_likelihood_file(path: str):
        with open(f"{path}.txt", "r") as file:
            lines = file.readlines()
            res = []
            for line in lines:
                res.append(list(map(float, line.strip().split(" "))))

            return res

    @staticmethod
    def parse_training_file(path: str):
        with open(f"{path}.txt", "r") as file:
            lines = file.readlines()
        file.close()
        result = []

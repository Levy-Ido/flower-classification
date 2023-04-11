import numpy as np
import matplotlib.pyplot as plt


def generate_sample():
    """
    Generate a 2x2 numpy array representing a sample with two binary features:
    whether the patient has cancer and whether they have a mutated gene.

    Returns:
        numpy.ndarray: A 2x2 numpy array where the first row represents the presence
        or absence of cancer and the second row represents the presence or absence
        of the mutated gene.
    """
    has_cancer = np.random.choice([True, False], p=[0.08, 0.92])
    has_mutated_gene = np.random.choice([True, False], p=[0.39, 0.61])
    return np.array([[has_cancer, not has_cancer], [has_mutated_gene, not has_mutated_gene]])


def calculate_odds_ratio():
    """
    Calculate the odds ratio of having cancer given the presence of a mutated gene
    based on a sample of patients.

    Returns:
        float: The odds ratio of having cancer given the presence of a mutated gene.
    """
    sample_size = np.random.randint(300, 401)
    matrix = np.sum([generate_sample() for _ in range(sample_size)], axis=0)
    cancer_odds = matrix[0][0] / matrix[0][1]
    mutated_gene_odds = matrix[1][0] / matrix[1][1]
    return cancer_odds / mutated_gene_odds


def build_vector():
    """
    Build a vector of 10,000 odds ratios of having cancer given the presence of a mutated gene.

    Returns:
        numpy.ndarray: A numpy array of length 10,000 representing the odds ratios of having
        cancer given the presence of a mutated gene.
    """
    vector = [calculate_odds_ratio() for _ in range(10_000)]
    return np.log(np.array(vector))


def generate_histogram():
    """
    Generate a histogram of the odds ratios of having cancer given the presence of a mutated gene.
    """
    plt.hist(build_vector(), bins=50)
    plt.xlabel('ln(odds)')
    plt.ylabel('count')
    plt.show()


if __name__ == "__main__":
    generate_histogram()
from collections import Counter


if __name__ == '__main__':
    with open("datasets/trial/tsar2022_en_trial_gold.tsv") as f:
        lines = f.readlines()

    texts = []
    target_word = []
    alternative_suggestions = []

    for line in lines:
        split_line = line.strip("\n").split("\t")
        texts.append(split_line[0])
        target_word.append(split_line[1])
        alternative_suggestions.append(Counter(split_line[2:]))



from collections import Counter


if __name__ == '__main__':
    with open("datasets/trial/tsar2022_es_trial_gold.tsv") as f:
        lines = f.readlines()

    texts = []
    target_word = []
    alternative_suggestions = []

    for line in lines:
        split_line = line.strip("\n").split("\t")
        texts.append(split_line[0])
        target_word.append(split_line[1])
        alternative_suggestions.append(Counter(split_line[2:]))

    for complex_word, context, alternatives in zip(target_word, texts, alternative_suggestions):
        print(complex_word)
        print(context)
        output_format = ""
        count = 1
        for alternative in alternatives:
            output_format += f"{count}. {alternative}\\n"
            count += 1
        print(output_format)
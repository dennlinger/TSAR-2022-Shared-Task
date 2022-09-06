"""
Uses a long context prediction setting for GPT-3.
"""

import regex
from typing import List

import openai

from config import API_KEY


def clean_predictions(text: str, given_word: str) -> List[str]:
    """
    Post-processing of files, by trying different strategies to coerce it into actual singular predictions.
    :param text: Unfiltered text predicted by a language model
    :return: List of individual predictions
    """
    # Presence of newlines within the prediction indicates prediction as list
    if "\n" in text.strip("\n "):
        cleaned_predictions = text.strip("\n ").split("\n")

    # Other common format contained comma-separated list without anything else
    elif "," in text.strip("\n "):
        cleaned_predictions = [pred.strip(" ") for pred in text.strip("\n ").split(",")]

    # Sometimes in-line enumerations also occur, this is a quick check to more or less guarantee
    # at least 6 enumerated predictions
    elif "1." in text and "6." in text:
        cleaned_predictions = regex.split(r"[0-9]{1,2}\.?", text.strip("\n "))

    else:
        raise ValueError(f"Unrecognized list format in prediction '{text}'")

    # Edge case where there is inconsistent newlines
    if 2 < len(cleaned_predictions) < 5:
        raise ValueError(f"Inconsistent newline pattern found in prediction '{text}'")

    # Remove numerals
    cleaned_predictions = [remove_numerals(pred) for pred in cleaned_predictions]
    # Make sure everything is lower-cased and stripped
    cleaned_predictions = [pred.lower().strip(" \n") for pred in cleaned_predictions]
    # Remove "to" in the beginning
    cleaned_predictions = [remove_to(pred) for pred in cleaned_predictions]
    # Remove predictions that match the given word
    cleaned_predictions = [pred for pred in cleaned_predictions if pred != given_word]
    # Remove empty predictions that may have slipped through:
    cleaned_predictions = [pred for pred in cleaned_predictions if pred.strip("\n ")]

    return cleaned_predictions


def remove_numerals(text: str) -> str:
    """
    Will remove any leading numerals (optionally with a dot).
    :param text: Input text, potentially containing a leading numeral
    :return: cleaned text
    """

    return regex.sub(r"[0-9]{1,2}\.? ?", "", text)


def remove_to(text: str) -> str:
    """
    Removes the leading "to"-infinitive from a prediction, which is sometimes caused when the context word
    is preceeded with a "to" in the text.
    :param text: Prediction text
    :return: Text where a leading "to " would be removed from the string.
    """
    return regex.sub(r"^to ", "", text)


if __name__ == '__main__':
    with open("datasets/trial/tsar2022_en_trial_none.tsv") as f:
        lines = f.readlines()

    openai.api_key = API_KEY
    for line in lines:
        context, word = line.strip("\n ").split("\t")
        prompt = f"Context: {context}\n" \
                 f"Question: Given the above context, list ten alternative words for \"{word}\" that are easier to understand.\n"\
                 f"Answer:"

        response = openai.Completion.create(
            model="text-davinci-002",
            prompt=prompt,
            stream=False,
            temperature=0.8,
            max_tokens=256,
            top_p=1,
            best_of=1,
            frequency_penalty=0.5,
            presence_penalty=0.3
        )
        predictions = response["choices"][0]["text"]

        predictions = clean_predictions(predictions, word)
        print(f"Complex word: {word}")
        print(predictions)
        # break

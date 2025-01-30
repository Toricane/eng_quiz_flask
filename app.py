# app.py
import os
import random
from dataclasses import dataclass
from typing import Dict, List

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from groq import Groq
from Levenshtein import distance, ratio

app = Flask(__name__)
load_dotenv()


@dataclass
class ReviewItem:
    question: str
    answer: str
    correct_count: int = 0


class VocabularyQuiz:
    def __init__(self):
        self.definitions = self.read_definitions("input.txt")
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    @staticmethod
    def clean_input(text: str) -> str:
        text = text.strip()
        text = "".join(c for c in text if c.isalpha() or c in "';")
        return text

    @staticmethod
    def read_definitions(filename: str) -> Dict[str, List[str]]:
        definitions = {}
        current_word = None

        with open(filename, "r") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue

                if line[0].isdigit():
                    parts = line.split(": ", 1)
                    word = parts[0].split(". ", 1)[1].strip()
                    definition = parts[1].strip()
                    definitions[word] = [definition]
                    current_word = word
                elif line.startswith("-"):
                    if current_word:
                        definition = line[1:].strip()
                        definitions[current_word].append(definition)

        return definitions

    def check_spelling(
        self, user_input: str, correct_word: str
    ) -> tuple[bool, int, float]:
        user_input = user_input.lower().strip()
        correct_word = correct_word.lower().strip()

        if user_input == correct_word:
            return True, 0, 1.0

        edit_distance = distance(user_input, correct_word)
        similarity = ratio(user_input, correct_word)

        return False, edit_distance, similarity

    def check_definition(
        self, user_definition: str, correct_definitions: List[str], word: str
    ) -> bool:
        system_prompt = (
            "You are a **strict** definition comparison assistant. Compare the user's definition "
            "with the correct definition(s) and respond with only 'yes' or 'no'."
        )

        user_message = (
            f"Word: {word}\n"
            f"User's definition: {user_definition}\n"
            f"Correct definition(s):\n"
            + "\n".join(f"- {d}" for d in correct_definitions)
        )

        completion = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0,
            max_tokens=1,
            top_p=1,
            stream=False,
            stop=None,
        )

        response = completion.choices[0].message.content.lower().strip()
        return response == "yes"

    def check_sentence(
        self, sentence: str, word: str, definitions: List[str]
    ) -> tuple[bool, str]:
        system_prompt = (
            "You are a very strict sentence evaluation assistant. Your job is to:\n"
            "1. Determine if the word is used correctly AND if the sentence provides sufficient context "
            "to demonstrate understanding of the word's meaning\n"
            "2. The sentence must contain enough context that someone who doesn't know the word could "
            "infer its meaning from how it's used\n"
            "3. Reject sentences that:\n"
            "   - Simply insert the word without meaningful context\n"
            "   - Use the word in a way that doesn't clearly demonstrate its meaning\n"
            "   - Are too vague or generic\n"
            "4. If usage is incorrect, provide an example sentence that clearly demonstrates "
            "the word's meaning through context\n\n"
            "For correct usage, respond with 'CORRECT'. For incorrect usage, respond with 'INCORRECT: ' "
            "followed by your example sentence that shows proper usage with clear context."
        )

        user_message = (
            f"Word: {word}\n"
            f"Definitions:\n"
            + "\n".join(f"- {d}" for d in definitions)
            + f"\nUser's sentence: {sentence}\n\n"
            "Remember: The sentence must provide enough context that someone unfamiliar with the "
            f"word could infer its meaning. Simple sentences like 'He showed {word}' or 'It was {word}' "
            "are not acceptable."
        )

        completion = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.7,
            max_tokens=100,
            top_p=1,
            stream=False,
            stop=None,
        )

        response = completion.choices[0].message.content.strip()

        if response.startswith("CORRECT"):
            return True, ""
        else:
            example_sentence = response.split("INCORRECT: ", 1)[1]
            return False, example_sentence


quiz = VocabularyQuiz()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/get_question/<quiz_type>")
def get_question(quiz_type):
    if quiz_type == "spelling":
        definitions = [
            (def_, word) for word, defs in quiz.definitions.items() for def_ in defs
        ]
        definition, word = random.choice(definitions)
        return jsonify({"question": definition, "answer": word, "type": "spelling"})
    elif quiz_type == "definition":
        word, definitions = random.choice(list(quiz.definitions.items()))
        return jsonify(
            {"question": word, "answer": definitions[0], "type": "definition"}
        )
    elif quiz_type == "sentence":
        word, definitions = random.choice(list(quiz.definitions.items()))
        return jsonify(
            {"question": word, "definitions": definitions, "type": "sentence"}
        )


@app.route("/check_answer", methods=["POST"])
def check_answer():
    data = request.json
    quiz_type = data["type"]
    user_answer = data["answer"]

    if quiz_type == "spelling":
        correct_word = data["correct_answer"]
        is_correct, _, similarity = quiz.check_spelling(user_answer, correct_word)
        return jsonify(
            {
                "correct": is_correct,
                "similarity": f"{similarity:.1%}",
                "correct_answer": correct_word,
            }
        )
    elif quiz_type == "definition":
        word = data["question"]
        correct_definitions = quiz.definitions[word]
        is_correct = quiz.check_definition(user_answer, correct_definitions, word)
        return jsonify(
            {"correct": is_correct, "correct_answer": correct_definitions[0]}
        )
    elif quiz_type == "sentence":
        word = data["question"]
        definitions = quiz.definitions[word]
        is_correct, example = quiz.check_sentence(user_answer, word, definitions)
        return jsonify(
            {"correct": is_correct, "example": example if not is_correct else ""}
        )


if __name__ == "__main__":
    app.run(debug=True)

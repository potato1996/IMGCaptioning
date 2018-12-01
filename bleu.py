import os
import sys

sys.path.append('../coco-caption')
from pycocoevalcap.bleu import Bleu


def score(ref, hypo):
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
    ]
    final_scores = {}
    for scorer, method in scorers:
        score, scores = scorer.compute_score(ref, hypo)
        for m, s in zip(method, score):
            final_scores[m] = s

    return final_scores


def evaluate(refs, res):
    """
    DEFINE sentence string
    :param refs: List[ID: Map(Int -> List(sentence))]
    :param res: List[ID: Map(Int -> List(sentence))]
    :return: Map("BLEU N": String -> score: float)
    """
    # compute bleu score
    final_scores = score(refs, res)

    # print out scores
    print('Bleu_1:\t', final_scores['Bleu_1'])
    print('Bleu_2:\t', final_scores['Bleu_2'])
    print('Bleu_3:\t', final_scores['Bleu_3'])
    print('Bleu_4:\t', final_scores['Bleu_4'])

    return final_scores

# reference = {1: ["this is small test and and and and and and"]}
# candidate = {1: ['this is a test and and and and']}
# score = evaluate(reference, candidate)
# print(score)

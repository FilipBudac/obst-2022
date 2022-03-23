from dataclasses import dataclass

import numpy as np
import pandas as pd

BOUNDARY = 50_000


@dataclass
class Node:
    value: str
    index: int


def calc_p_closure(total):
    def calc_p(frequency):
        return frequency / total
    return calc_p


def calc_q_closure(sorted_words, total):
    def calc_q(prev, cur):
        return sum(f for f, _ in sorted_words[prev:cur]) / total
    return calc_q


def calc_tables(s_words_matrix, calc_p, calc_q):
    p, q = [], []

    prev = -1
    for cur, (f, word) in enumerate(s_words_matrix):
        if f > BOUNDARY:
            p.append(calc_p(f))
            q.append(calc_q(prev + 1, cur))
            prev = cur

    if prev < len(s_words_matrix) - 1:
        q.append(calc_q(prev + 1, len(s_words_matrix)))

    return p, q


def const_obst(p, q, keys):
    n = len(p)
    p = pd.Series(p, index=range(1, n + 1))
    q = pd.Series(q)

    e = pd.DataFrame(np.diag(q), index=range(1, n + 2))
    w = pd.DataFrame(np.diag(q), index=range(1, n + 2))
    root = pd.DataFrame(np.zeros((n, n)), index=range(1, n + 1), columns=range(1, n + 1))

    for l in range(1, n + 1):
        for i in range(1, n - l + 2):
            j = i + l - 1
            e.at[i, j] = np.inf
            w.at[i, j] = w.at[i, j - 1] + p[j] + q[j]
            for r in range(i, j + 1):
                t = e.at[i, r - 1] + e.at[r + 1, j] + w.at[i, j]
                if t < e.at[i, j]:
                    e.at[i, j] = t
                    root.at[i, j] = Node(value=keys[r - 1], index=r)
    return root


def find_word_in_obst_rec(root, word, r, c, depth):
    node = root.at[r, c]
    if node.value == word:
        return depth
    if word > node.value:
        return find_word_in_obst_rec(root, word, node.index + 1, c, depth + 1)
    if word < node.value:
        return find_word_in_obst_rec(root, word, r, node.index - 1, depth + 1)


def find_word_in_obst_closure(root):
    def find_word_in_obst(word):
        depth, r, c = 1, 1, len(root)
        return find_word_in_obst_rec(root, word, r, c, depth)
    return find_word_in_obst


def main():
    with open('input.txt') as file:
        words_matrix = [(int(line.split(' ')[0]), line.rstrip().split(' ')[1]) for line in file]

    total = sum(f for f, _ in words_matrix)

    words_matrix = sorted(words_matrix,  key=lambda pair: pair[1])
    keys = sorted([word for f, word in words_matrix if f > BOUNDARY], key=lambda word: word)

    cal_p, cal_q = calc_p_closure(total), calc_q_closure(words_matrix, total)
    p, q = calc_tables(words_matrix, cal_p, cal_q)

    root = const_obst(p, q, keys)
    find_word_in_obst = find_word_in_obst_closure(root)

    words = ['people', 'of', 'the', 'for', 'must', 'again', 'going', 'time', 'like', 'very', 'will', 'each', 'it', 'up',
             'and', 'could', 'make', 'to', 'said', 'about']
    for word in words:
        print(f'{word}: {find_word_in_obst(word)}')


if __name__ == '__main__':
    main()

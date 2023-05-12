#! /usr/bin/python

import sys
from collections import defaultdict
import math
from collections import Counter
from itertools import product
import re

"""
Count n-gram frequencies in a data file and write counts to
stdout. 
"""


def simple_conll_corpus_iterator(corpus_file):
    """
    Get an iterator object over the corpus file. The elements of the
    iterator contain (word, ne_tag) tuples. Blank lines, indicating
    sentence boundaries return (None, None).
    """
    l = corpus_file.readline()
    while l:
        line = l.strip()
        if line:  # Nonempty line
            # Extract information from line.
            # Each line has the format
            # word pos_tag phrase_tag ne_tag
            fields = line.split(" ")
            ne_tag = fields[-1]
            # phrase_tag = fields[-2] #Unused
            # pos_tag = fields[-3] #Unused
            word = " ".join(fields[:-1])
            yield word, ne_tag
        else:  # Empty line
            yield (None, None)
        l = corpus_file.readline()


def sentence_iterator(corpus_iterator):
    """
    Return an iterator object that yields one sentence at a time.
    Sentences are represented as lists of (word, ne_tag) tuples.
    """
    current_sentence = []  # Buffer for the current sentence
    for l in corpus_iterator:
        if l == (None, None):
            if current_sentence:  # Reached the end of a sentence
                yield current_sentence
                current_sentence = []  # Reset buffer
            else:  # Got empty input stream
                sys.stderr.write("WARNING: Got empty input file/stream.\n")
                raise StopIteration
        else:
            current_sentence.append(l)  # Add token to the buffer

    if current_sentence:  # If the last line was blank, we're done
        yield current_sentence  # Otherwise when there is no more token
        # in the stream return the last sentence.


def get_ngrams(sent_iterator, n):
    """
    Get a generator that returns n-grams over the entire corpus,
    respecting sentence boundaries and inserting boundary tokens.
    Sent_iterator is a generator object whose elements are lists
    of tokens.
    """
    for sent in sent_iterator:
        # Add boundary symbols to the sentence
        w_boundary = (n - 1) * [(None, "*")]
        w_boundary.extend(sent)
        w_boundary.append((None, "STOP"))
        # Then extract n-grams
        ngrams = (tuple(w_boundary[i : i + n]) for i in range(len(w_boundary) - n + 1))
        for n_gram in ngrams:  # Return one n-gram at a time
            yield n_gram


class Hmm(object):
    """
    Stores counts for n-grams and emissions.
    """

    def __init__(self, n=3):
        assert n >= 2, "Expecting n>=2."
        self.n = n
        self.emission_counts = defaultdict(int)
        self.ngram_counts = [defaultdict(int) for i in range(self.n)]
        self.all_states = set()
        self.c = Counter()  # contains all words except rare
        self.s = set()  # contains rare words
        self.e = defaultdict(float)
        self.q = defaultdict(float)

    def preprocess(self, corpus_file):
        """Preprocess the train file to categorize rare words into 4 categories"""
        c = Counter()
        with open(corpus_file, "r") as file:
            lines = file.readlines()
            for x in lines:
                if x.strip():
                    a = x.strip().split(" ")
                    c[a[0]] += 1
        for x in c:
            if c[x] < 5:
                self.s.add(x)
            else:
                self.c[x] = c[x]

        with open("modified_train", "w") as f:
            with open(corpus_file, "r") as file:
                l = file.readlines()
                for x in l:
                    if x.strip():
                        a = x.strip().split(" ")
                        if a[0] in self.s:
                            new_tag = self.categorize(a[0])
                            f.write(new_tag + " " + a[1] + "\n")
                        else:
                            f.write(x)
                    else:
                        f.write(x)

    def train(self, corpus_file):
        """
        Count n-gram frequencies and emission probabilities from a corpus file.
        """
        ngram_iterator = get_ngrams(
            sentence_iterator(simple_conll_corpus_iterator(corpus_file)), self.n
        )

        for ngram in ngram_iterator:
            # Sanity check: n-gram we get from the corpus stream needs to have the right length
            assert len(ngram) == self.n, "ngram in stream is %i, expected %i" % (
                len(ngram, self.n)
            )

            tagsonly = tuple(
                [ne_tag for word, ne_tag in ngram]
            )  # retrieve only the tags
            for i in range(2, self.n + 1):  # Count NE-tag 2-grams..n-grams
                self.ngram_counts[i - 1][tagsonly[-i:]] += 1

            if ngram[-1][0] is not None:  # If this is not the last word in a sentence
                self.ngram_counts[0][tagsonly[-1:]] += 1  # count 1-gram
                self.emission_counts[ngram[-1]] += 1  # and emission frequencies

            # Need to count a single n-1-gram of sentence start symbols per sentence
            if ngram[-2][0] is None:  # this is the first n-gram in a sentence
                self.ngram_counts[self.n - 2][tuple((self.n - 1) * ["*"])] += 1

    def write_counts(self, output, printngrams=[1, 2, 3]):
        """
        Writes counts to the output file object.
        Format:

        """
        # First write counts for emissions
        for word, ne_tag in self.emission_counts:
            output.write(
                "%i WORDTAG %s %s\n"
                % (self.emission_counts[(word, ne_tag)], ne_tag, word)
            )

        # Then write counts for all ngrams
        for n in printngrams:
            for ngram in self.ngram_counts[n - 1]:
                ngramstr = " ".join(ngram)
                output.write(
                    "%i %i-GRAM %s\n" % (self.ngram_counts[n - 1][ngram], n, ngramstr)
                )

    def read_counts(self, corpusfile):
        """Adds n-gram counts to the dictionary"""

        self.n = 3
        self.emission_counts = defaultdict(int)
        self.ngram_counts = [defaultdict(int) for i in xrange(self.n)]
        self.all_states = set()

        for line in corpusfile:
            parts = line.strip().split(" ")
            count = float(parts[0])
            if parts[1] == "WORDTAG":
                ne_tag = parts[2]
                word = parts[3]
                self.emission_counts[(word, ne_tag)] = count
                self.all_states.add(ne_tag)
            elif parts[1].endswith("GRAM"):
                n = int(parts[1].replace("-GRAM", ""))
                ngram = tuple(parts[2:])
                self.ngram_counts[n - 1][ngram] = count

    def emissionprobability(self):
        """Gives the emission probability"""
        for word, tag in self.emission_counts:
            self.e[(word, tag)] = (
                self.emission_counts[(word, tag)] / self.ngram_counts[0][(tag,)]
            )

    def transitionprobability(self):
        """Gives the transition probability"""
        for x, y, z in self.ngram_counts[2]:
            self.q[(x, y, z)] = (
                self.ngram_counts[2][(x, y, z)] / self.ngram_counts[1][(x, y)]
            )

    def categorize(self, word):
        """fUNCTION TO SUBCATEGORIZE WORDS INTO 4 CATEGORIES"""
        import string

        # check for a number
        if word.isnumeric():
            return "_NUM_"
        # check if all letters are capital
        elif re.findall(r"([A-Z]+)$", word):
            return "_ALLCAPS_"
        elif word in string.punctuation:
            return "_PUNCT_"
        else:
            return "_RARE_"

    def base_tagger(self, filename):
        """Unigram Model"""
        f = open("gene_dev.p1.out", "w")
        with open(filename, "r") as file:
            l = file.readlines()
            for x in l:
                if x.strip():
                    a = x.strip()
                    m = 0
                    if a not in self.c:
                        new_word = self.categorize(a)
                        for word, tag in self.emission_counts:
                            if new_word == word:
                                m = max(
                                    (
                                        self.emission_counts[(word, tag)]
                                        / self.ngram_counts[0][(tag,)]
                                    ),
                                    m,
                                )
                                if m == (
                                    self.emission_counts[(word, tag)]
                                    / self.ngram_counts[0][(tag,)]
                                ):
                                    f_tag = tag
                        f.write(a + " " + f_tag + "\n")
                    else:
                        for word, tag in self.emission_counts:
                            if a == word:
                                m = max(
                                    (
                                        self.emission_counts[(a, tag)]
                                        / self.ngram_counts[0][(tag,)]
                                    ),
                                    m,
                                )
                                if m == (
                                    self.emission_counts[(a, tag)]
                                    / self.ngram_counts[0][(tag,)]
                                ):
                                    f_tag = tag
                        f.write(a + " " + f_tag + "\n")
                else:
                    f.write(x)

    def viterbi(self, filename):
        """Trigram HMM model using viterbi algorithm"""
        f = open("gene_dev.viterbi", "w")
        file = open(filename, "r")
        sen_iterator = sentence_iterator(simple_conll_corpus_iterator(file))
        sentences = []
        pi = {}
        pi[(0, "*", "*")] = 1
        bp = {}
        for x in sen_iterator:
            a = []
            for y in x:
                a.append(y[1])
            sentences.append(a)
        corr_tags = []

        for x in sentences:
            n = len(x)
            tags = [None] * n
            for k in range(1, n + 1):
                word = x[k - 1]
                if x[k - 1] not in self.c:
                    word = self.categorize(x[k - 1])
                if k == 1:
                    pi[(1, "*", "O")] = (
                        pi[(0, "*", "*")]
                        * self.q[("*", "*", "O")]
                        * self.e[(word, "O")]
                    )
                    bp[(1, "*", "O")] = "O"
                    pi[(1, "*", "I-GENE")] = (
                        pi[(0, "*", "*")]
                        * self.q[("*", "*", "I-GENE")]
                        * self.e[(word, "I-GENE")]
                    )
                    bp[(1, "*", "I-GENE")] = "I-GENE"

                elif k == 2:
                    for a in list(product(["O", "I-GENE"], repeat=2)):
                        pi[(2, a[0], a[1])] = (
                            pi[(1, "*", a[0])]
                            * self.q[("*", a[0], a[1])]
                            * self.e[(word, a[1])]
                        )
                        bp[(2, a[0], a[1])] = a[1]

                else:
                    for a in list(product(["O", "I-GENE"], repeat=2)):
                        pi[(k, a[0], a[1])] = max(
                            pi[(k - 1, "I-GENE", a[0])]
                            * self.q[("I-GENE", a[0], a[1])]
                            * self.e[(word, a[1])],
                            pi[(k - 1, "O", a[0])]
                            * self.q[("O", a[0], a[1])]
                            * self.e[(word, a[1])],
                        )
                        if (
                            pi[(k, a[0], a[1])]
                            == pi[(k - 1, "I-GENE", a[0])]
                            * self.q[("I-GENE", a[0], a[1])]
                            * self.e[(word, a[1])]
                        ):
                            bp[(k, a[0], a[1])] = "I-GENE"
                        else:
                            bp[(k, a[0], a[1])] = "O"

            m = 0
            for a in list(product(["O", "I-GENE"], repeat=2)):
                m = max(pi[(n, a[0], a[1])] * self.q[a[0], a[1], "STOP"], m)
                if m == pi[(n, a[0], a[1])] * self.q[a[0], a[1], "STOP"]:
                    tags[n - 2], tags[n - 1] = a[0], a[1]
            for k in range(n - 3, -1, -1):
                tags[k] = bp[(k + 3, tags[k + 1], tags[k + 2])]

            for a, b in zip(x, tags):
                f.write(a + " " + b + "\n")
            f.write("\n")


def usage():
    print(
        """
    python count_freqs.py [input_file] > [output_file]
        Read in a gene tagged training input file and produce counts.
    """
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:  # Expect exactly one argument: the training data file
        usage()
        sys.exit(2)

    try:
        input = sys.argv[1]
    except IOError:
        sys.stderr.write("ERROR: Cannot read inputfile %s.\n" % arg)
        sys.exit(1)

    # Initialize a trigram counter
    counter = Hmm(3)
    # Collect counts
    counter.preprocess(input)
    new_input = open("modified_train", "r")
    counter.train(new_input)
    counter.base_tagger("gene.dev")
    counter.transitionprobability()
    counter.emissionprobability()
    counter.viterbi("gene.dev")
    # Write the counts
    # counter.write_counts(sys.stdout)

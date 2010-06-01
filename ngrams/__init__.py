#!/usr/bin/env python
# -*- coding: utf-8 -*-
u"""
# -*- coding: utf-8 -*-
(c) 2009 Ryszard Szopa <ryszard.szopa@gmail.com>
(c) 2010 Mahmoud Abdelkader <mahmoud@linux.com>

This work 'as-is' we provide.
No warranty, express or implied.
We’ve done our best,
to debug and test.
Liability for damages denied.

Permission is granted hereby,
to copy, share, and modify.
Use as is fit,
free or for profit.
On this notice these rights rely.

This library provides two classes: Ngrams, which treats words as tokens

    >>> (Ngrams(u'This is a very small donkey.') *
    ...  Ngrams(u'This animal is a very small donkey.')) #doctest:+ELLIPSIS
    0.6708203932...

and CharNgrams, which treats single characters as tokens:

    >>> (CharNgrams('supercalifragilistic') *
    ...  CharNgrams('supercalifragislislistic')) #doctest:+ELLIPSIS
    0.757...

If none of these fits your definition of `token' all you have to do is
subclass Ngrams and define you own tokenize method.

When creating an Ngrams object you can provide a second argument as
the value of n (the default being 3). You can compare only n-grams
with the same value of n.

    >>> Ngrams('ala ma kota', 3) * Ngrams('ala ma kota', 2)
    Traceback (most recent call last):
      ...
    WrongN

"""
import re
import math
from collections import defaultdict
from itertools import tee, izip


def n_wise(iterable, n):
    """Returns n iterators for an iterable that are sequentially
    n-wise

    """
    n_iterators = tee(iterable, n)
    zippables = [n_iterators[0]]

    for advance, iteratee in enumerate(n_iterators[1:]):
        advance += 1  # since enumerate is 0 indexed.
        while advance > 0:
            # we advance the iterator `advance+1` steps
            next(iteratee, None)
            advance -= 1
        # append everything to the zippables
        zippables.append(iteratee)
    # return the izip expansion of each iterator
    return izip(*zippables)


class Ngrams(object):
    """Compare strings using an n-grams model and cosine similarity.

    This class uses words as tokens. See module docs.

    >>> import pprint
    >>> items = sorted(Ngrams('Compare strings using an n-grams model and '
    ...                       'cosine similarity. This class uses words as'
    ...                       'tokens. See module docs.').d.items())
    >>> pprint.pprint(items)
    [('an ngrams model', 0.2581988897471611),
     ('and cosine similarity', 0.2581988897471611),
     ('astokens see module', 0.2581988897471611),
     ('class uses words', 0.2581988897471611),
     ('compare strings using', 0.2581988897471611),
     ('cosine similarity this', 0.2581988897471611),
     ('model and cosine', 0.2581988897471611),
     ('ngrams model and', 0.2581988897471611),
     ('see module docs', 0.2581988897471611),
     ('similarity this class', 0.2581988897471611),
     ('strings using an', 0.2581988897471611),
     ('this class uses', 0.2581988897471611),
     ('uses words astokens', 0.2581988897471611),
     ('using an ngrams', 0.2581988897471611),
     ('words astokens see', 0.2581988897471611)]

    """

    ngram_joiner = " "

    class WrongN(Exception):
        """Error to raise when two ngrams of different n's are being
        compared.
        """
        pass

    def __init__(self, text, n=3):
        self.n = n
        self.text = text
        self.d = self.text_to_ngrams(self.text)

    def __getitem__(self, word):
        return self.d[word]

    def __contains__(self, word):
        return word in self.d

    def __iter__(self):
        return iter(self.d)

    def __mul__(self, other):
        """Returns the similarity between self and other as a float in
        (0;1).
        """
        if self.n != other.n:
            raise self.WrongN

        score = 0
        if self.text == other.text:
            score = 1.0
        else:
            score = sum(self[k] * other[k] for k in self if k in other)

        return score

    def __repr__(self):
        return "Ngrams(%r, %r)" % (self.text, self.n)

    def __str__(self):
        return self.text

    def tokenize(self, text):
        """Return a sequence of tokens from which the ngrams should be
        constructed.

        This shouldn't be a generator, because its length will be
        needed.
        """

        regex = re.compile(u'[^\w\n ]|\xe2', re.UNICODE)
        return regex.sub('', text).lower().split()

    def relative_n_gram_dist(self, other):
        """Generates the relative n-gram distance between two strings.
        The relative n-gram distance can be used as a basis function to
        determine the similarity of strings.

        The smaller the relative n-gram distance is, the similar the
        two strings are.

        The formula is:
                                 |nGram(s1) INTERSECT nGram(s2)|
         Delta_q(s1, s2) = 1  -  -------------------------------
                                 |nGram(s1)   UNION   nGram(s2)|
        """
        ngrams = set(self.d.keys())
        other_ngrams = set(other.d.keys())
        result = None
        try:
            result = (1 - (abs(len(ngrams.intersection(other_ngrams))) /
                           float(abs(len(ngrams.union(other_ngrams))))))
        except ZeroDivisionError:
            # I don't know, is there a more graceful way to handle this?
            raise

        return result

    def normalize(self, text):
        """This method is run on the text right before tokenization"""
        try:
            return text.lower()
        except AttributeError:
            # text is not a string?
            raise TypeError(text)

    def make_ngrams(self, text):
        """
        # -*- coding: utf-8 -*-
        Return an iterator of tokens of which the n-grams will
        consist. You can overwrite this method in subclasses.

        >>> import pprint
        >>> L = list(Ngrams('').make_ngrams(chr(10).join([
        ...         u'This work \\'as-is\\' we provide.',
        ...         u'No warranty, express or implied.',
        ...         u'We\\'ve done our best,',
        ...         u'to debug and test.',
        ...         u'Liability for damages denied.'])))[:5]
        >>> pprint.pprint(L)
        [u'this work asis',
         u'work asis we',
         u'asis we provide',
         u'we provide no',
         u'provide no warranty']

        """

        text = self.normalize(text)
        tokens = self.tokenize(text)
        return (self.ngram_joiner.join(i) for i in n_wise(tokens, self.n))

    def text_to_ngrams(self, text):
        counts = defaultdict(int)

        for ngram in self.make_ngrams(text):
            counts[ngram] += 1

        norm = math.sqrt(sum(x ** 2 for x in counts.itervalues()))

        for k, v in counts.iteritems():
            counts[k] = v / norm

        return counts


class CharNgrams(Ngrams):

    """Ngrams comparison using single characters as tokens.

    >>> CharNgrams('ala ma kota')*CharNgrams('ala ma kota')
    1.0

    >>> round(CharNgrams('This Makes No Difference') *
    ...       CharNgrams('this makes no difference'), 4)
    1.0
    >>> (CharNgrams('Warszawska')*CharNgrams('Warszawa') >
    ...  CharNgrams('Warszawa')*CharNgrams('Wawa'))
    True

    """

    ngram_joiner = ""

    def tokenize(self, st):
        """
        >>> ''.join(CharNgrams('').tokenize('ala ma kota!'))
        'alamakota'
        """
        return [c for c in st if c.isalnum()]


class CharNgramSpaces(CharNgrams):
    """Like CharNgrams, except it keeps whitespace as one space in
    the process of tokenization. This should be useful for analyzing
    texts longer than words, where places at which word boundaries
    occur may be important.

    """
    def tokenize(self, st):
        new_st = re.sub(r'\s+', ' ', st)
        return [c for c in new_st if c.isalnum() or c == " "]


if __name__ == '__main__':
    import doctest
    doctest.testmod()


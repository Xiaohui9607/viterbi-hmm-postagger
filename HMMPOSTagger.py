from collections import OrderedDict
import numpy as np
import re

digit_regrex = r'\b-?\d+\:?\\?\/?\,?\.?\d*\,?\.?\d*\,?\.?\d*\b'
cap_regrex = r'(?<![\`\`\.\n])(\n[A-Z]+[\w\-\\\/]*\b)'
hypen3_regex = r'\b.*\-\w+\-\w+\b'
hypen2_regex = r'\b\w+\-\w+\b'
hypen2s_regex = r'\b\w+\-\w+s\b'
hypen2ing_regex = r'\b\w+\-\w+ing\b'

sf = ['ble', 'al', 'algia', 'an', 'ance', 'ancy', 'ant', 'tion', 'acity', 'el', 'ery'
    'ry','ate', 'cule', 'cy', 'dom', 'ee', 'en', 'ly', 'at', 'ac', 'ado', 'age','ard'
    'ence', 'ency', 'er', 'or', 'escent', 'ese', 'esis', 'osis', 'ess', 'et', 'artic'
    'ette', 'fic', 'ful', 'fy', 'hood', 'ic', 'ice', 'id', 'ide', 'ine', 'ion',
    'ish', 'ism', 'ist', 'ite', 'ty', 'ive', 'ize', 'less', 'em', 'in', 'like',
    'ment', 'ness', 'oid', 'logy', 's', 'ed', 'ing', 'est', 'un', 'ical', 'ics']

suffixes = OrderedDict(list(zip(sf, range(len(sf)))))

# for numerical statbility
alpha = 1e-7

def logdotexp(A, B):    # log matrix multiplication
    max_A = np.max(A)
    max_B = np.max(B)
    C = np.dot(np.exp(A - max_A), np.exp(B - max_B))
    np.log(C, out=C)
    C += max_A + max_B
    return C


class HMMPOS:
    def __init__(self):
        self.A_unigram = None
        self.A_bigram = None
        self.A_trigram = None
        self.B = None
        self.C = None
        self.vocab = None
        self.vocab_size = 0
        self.tag = None
        self.tag_size = 0
        self.pi_index = None
        self.suffixes = suffixes

    def fit(self, train_pos):
        words, tags = self._preprocess_train(train_pos) # preprocess text and split words and tags
        self._create_reprensentation(words, tags)       # create POS and VOCAB dictionary (look up table)
        uni_count, bi_count = self._estimate_A_unigram_bigram(tags) # estimate unigram and bigram transition matrix
        tri_count = self._estimate_A_trigram(tags)      # estimate trigram transition matrix
        self._estimate_B(words, tags)   # estimate token emission matrix
        self._estimate_C()              # estimate suffix emission matrix (used to process unknown word statiscally)
        self.norm_w = self._delete_interpolation(uni_count, bi_count, tri_count)    # weighting unigram, bigram and trigram
        # combine 3 matrixes together
        self.A_bigram = ((self.norm_w[1] * self.A_bigram).T + self.norm_w[0] * self.A_unigram).T
        self.A_trigram = (self.norm_w[2] * self.A_trigram.transpose([1, 0, 2]) + self.A_bigram).transpose([1, 0, 2])
        self.A_trigram = np.log(self.A_trigram)
        self.A_bigram = np.log(self.A_bigram)
        self.B = np.log(self.B+alpha)
        self.C = np.log(self.C+alpha)
        self.pi_index = self.tag['.']   # pi probabilities as indicated in the textbook

    def predict(self, test_words, outfile=None):
        pr_sentences, wr_sentences = self._preprocess_test(test_words)  # preprocess text, one for predicte, one for writing into file
        outs = []
        for sentence in pr_sentences:   # make prediction
            if sentence:
                outs.append(self._vertibi(sentence))
        if outfile:                     # write into file
            with open(outfile, 'w') as f:
                for pr_sentence, wr_sentence, tags in zip(pr_sentences, wr_sentences, outs):
                    for idx, (pr_word, wr_word, tag) in enumerate(zip(pr_sentence, wr_sentence, tags)):
                        tag = self._postprocess_unknown(pr_word, wr_word, tag)
                        f.write("%s\t%s\n" % (wr_word, tag))
                    f.write("\n")
                f.close()

    def _preprocess_test(self, test_words):
        text = open(test_words).read()
        pr_text = self._text_regex(text)
        pr_sentences = pr_text.split('\n\n')
        pr_sentences = [sentence.split() for sentence in pr_sentences]
        wr_sentences = text.split('\n\n')
        wr_sentences = [sentence.split() for sentence in wr_sentences]
        return pr_sentences, wr_sentences

    def _preprocess_train(self, train_pos):
        text = open(train_pos).read()
        text = self._text_regex(text)
        word_tags = [word_tag.split('\t') for word_tag in text.split('\n') if word_tag]
        words = [word_tag[0] for word_tag in word_tags]
        tags = [word_tag[1] for word_tag in word_tags]
        return words, tags

    def _text_regex(self, text):
        text = re.sub(cap_regrex, '\n$$cap$$', text)
        text = text.lower().strip()
        text = re.sub(digit_regrex, '$$digit$$', text)
        text = re.sub(hypen3_regex, '$$hypen3$$', text)
        text = re.sub(hypen2ing_regex, '$$hypen2ing$$', text)
        text = re.sub(hypen2s_regex, '$$hypen2s$$', text)
        text = re.sub(hypen2_regex, '$$hypen2$$', text)
        return text

    def _estimate_A_trigram(self, tags):
        count_numerator = np.ones((self.tag_size, self.tag_size, self.tag_size)) * alpha  # A -> [target, given1, given2]
        count_denominator = np.ones((self.tag_size, self.tag_size)) * self.tag_size ** 2 * alpha
        for t_i in range(len(tags)-2):
            t_given_1, t_given_2 = tags[t_i:t_i+2]
            t_target = tags[t_i+2]
            count_numerator[self.tag[t_target], self.tag[t_given_1], self.tag[t_given_2]] += 1
            count_denominator[self.tag[t_given_1], self.tag[t_given_2]] += 1
        self.A_trigram = count_numerator / count_denominator
        return count_numerator

    def _estimate_A_unigram_bigram(self, tags):
        count_numerator = np.ones((self.tag_size, self.tag_size)) * alpha    # A -> [target, given2]
        count_denominator = np.ones(self.tag_size) * self.tag_size * alpha
        for t_i in range(len(tags)-1):
            t_given = tags[t_i]
            t_target = tags[t_i+1]
            count_numerator[self.tag[t_target], self.tag[t_given]] += 1
            count_denominator[self.tag[t_given]] += 1
        self.A_unigram = count_denominator / np.sum(count_denominator)
        self.A_bigram = count_numerator / count_denominator
        return count_denominator, count_numerator

    def _estimate_B(self, words, tags):
        count_numerator = np.zeros((self.tag_size, self.vocab_size))    # B -> [tag, word]
        count_denominator = np.zeros(self.tag_size)
        for w_target, t_given in zip(words, tags):
            count_numerator[self.tag[t_given], self.vocab[w_target]] += 1
            count_denominator[self.tag[t_given]] += 1
        self.B = (count_numerator.T / count_denominator).T

    def _vertibi(self, sentence):
        V = np.zeros((self.tag_size, len(sentence)))            # probability table
        BP = np.zeros_like(V, dtype=np.int)                     # back pointer table
        emiss_prob = self.B[:, self.vocab[sentence[0]]] if sentence[0] in self.vocab else self._process_unknown(sentence[0])    # emission prob
        V[:, 0] = self.A_bigram[:, self.pi_index] + emiss_prob          # log(emission prob * transition prob)
        for w_i, word in enumerate(sentence[1:], start=1):
            emiss_prob = self.B[:, self.vocab[word]] if word in self.vocab else self._process_unknown(word)     # emission prob
            if w_i == 1:
                trans_prob = self.A_bigram + V[:, w_i-1]
            else:
                trans_prob = np.max(self.A_trigram + logdotexp(V[:, w_i-2:w_i-1], V[:, w_i-1:w_i].T), axis=1)
            t = np.max(trans_prob.T + emiss_prob, axis=0).T             # log(emission prob * transition prob)
            V[:, w_i] = t                                               # track tables
            BP[:, w_i] = np.argmax(trans_prob.T + emiss_prob, axis=0).T
        BPp = np.argmax(V[:, -1])
        path = []
        for i in range(len(sentence) - 1, -1, -1):                      # back track tables
            path.append(BPp)
            BPp = BP[BPp, i]
        path.reverse()
        return [self._map_index_to_tag(i) for i in path]

    def _create_reprensentation(self, words, tags):
        self.vocab_size = len(set(words))
        self.tag_size = len(set(tags))
        word_items = list(zip(set(words), range(self.vocab_size)))
        tag_items = list(zip(set(tags), range(self.tag_size)))
        self.vocab = OrderedDict(word_items)
        self.tag = OrderedDict(tag_items)

    def _map_index_to_tag(self, index):
        return list(self.tag.keys())[index].upper()

    def _delete_interpolation(self, uni_count, bi_count, tri_count):
        weight = np.zeros(3)
        for t_target in self.tag.values():
            for t_given_1 in self.tag.values():
                for t_given_2 in self.tag.values():
                    v = tri_count[t_given_2, t_given_1, t_target]
                    if v > 0:
                        try:
                            c1 = (v-1) / (bi_count[t_given_1, t_target]-1)
                        except ZeroDivisionError:
                            c1 = 0
                        try:
                            c2 = (bi_count[t_given_2, t_given_1]-1) / (uni_count[t_given_1]-1)
                        except ZeroDivisionError:
                            c2 = 0
                        try:
                            c3 = (uni_count[t_given_2]-1)/(np.sum(uni_count)-1)
                        except ZeroDivisionError:
                            c3 = 0
                        k = np.argmax([c1, c2, c3])
                        weight[2-k] += v
        weight /= np.sum(weight)
        return weight

    def _postprocess_unknown(self, pr_word, wr_word, tag):
        if wr_word not in self.vocab:
            if pr_word == '$$cap$$':
                tag = 'NNP'
                if wr_word.lower() in self.vocab:
                    if np.argmax(self.B[:, self.vocab[wr_word.lower()]]) == self.tag['nns']:
                        tag = 'NNPS'
            elif pr_word.endswith('es'):
                if pr_word[:-2] in self.vocab:
                    max_prob_tag = np.argmax(self.B[:, self.vocab[pr_word[:-2].lower()]])
                    if max_prob_tag == self.tag['nn']:
                        tag = 'NNS'
                    elif max_prob_tag == self.tag['vb']:
                        tag = 'VBZ'
            elif pr_word.endswith('s'):
                if pr_word[:-1] in self.vocab:
                    max_prob_tag = np.argmax(self.B[:, self.vocab[pr_word[:-1].lower()]])
                    if max_prob_tag == self.tag['nn']:
                        tag = 'NNS'
                    elif max_prob_tag == self.tag['vb']:
                        tag = 'VBZ'
            elif pr_word.endswith('ing'):
                if pr_word[:-3] in self.vocab:
                    max_prob_tag = np.argmax(self.B[:, self.vocab[pr_word[:-3].lower()]])
                    if max_prob_tag == self.tag['vb']:
                        tag = 'VBG'
            elif pr_word.endswith('ed'):
                if pr_word[:-2] in self.vocab:
                    max_prob_tag = np.argmax(self.B[:, self.vocab[pr_word[:-2].lower()]])
                    if max_prob_tag == self.tag['vb']:
                        tag = 'VBD'
        return tag

    def _estimate_C(self):
        self.C = np.zeros((self.tag_size, len(self.suffixes)))
        for word in self.vocab:
            suf = ''
            for k in self.suffixes:
                if word.endswith(k):
                    suf = k
            if suf:
                self.C[:, self.suffixes[suf]] += self.B[:, self.vocab[word]]
        self.C = (self.C.T / (np.sum(self.C.T, axis=0)+alpha)).T

    def _process_unknown(self, word):
        suf = ''
        for k in self.suffixes:
            if word.endswith(k):
                suf = k
        if suf:
            return self.C[:, self.suffixes[suf]]
        return np.zeros(self.tag_size)


if __name__ == '__main__':
    train_path = "./POS/POS_train.pos"
    test_words = './POS/POS_test.words'
    hmm = HMMPOS()
    hmm.fit(train_pos=train_path)
    hmm.predict(test_words=test_words, outfile="POS_test.pos")



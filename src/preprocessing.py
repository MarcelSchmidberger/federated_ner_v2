import torch
import numpy as np
import time
import gensim

def read_files(filenames, *, encoding="UTF8"):
    sentences = []
    for filename in filenames:
        with open(filename, mode='rt', encoding=encoding) as f:
            sentence = []
            for line in f:
                if len(line) == 0 or line.startswith('-DOCSTART') or line[0] == "\n":
                    if len(sentence) > 0:
                        sentences.append(sentence)
                        sentence = []
                    continue
                splits = line.split(' ')
                sentence.append([splits[0], splits[-1]])

            if len(sentence) > 0:
                sentences.append(sentence)
                sentence = []

    return sentences

def extract_words_and_labels(sentences):
    labels = set()
    words = set()

    print("Extracting words and labels...")
    for sentence in sentences:
        for token, label in sentence:
            labels.add(label)
            words.add(token.lower())
    print(f"Extracted {len(words)} words and {len(labels)} labels.")

    return words, labels

def prepare_indices(sentences):
    words, labels = extract_words_and_labels(sentences)

    # mapping for words
    word2Idx = {}
    word2Idx["PADDING_TOKEN"] = 0
    word2Idx["UNKNOWN_TOKEN"] = 1

    for word in words:
        word2Idx[word] = len(word2Idx)

    # mapping for labels
    label2Idx = {}
    for label in labels:
        label2Idx[label] = len(label2Idx)
    
    idx2Label = {v: k for k, v in label2Idx.items()}

    return word2Idx, label2Idx, idx2Label

def prepare_embeddings(sentences, embeddings_path):
    words, labels = extract_words_and_labels(sentences)

    label2Idx = {}
    for label in labels:
        label2Idx[label] = len(label2Idx)
    
    idx2Label = {v: k for k, v in label2Idx.items()}

    word2Idx = {} 
    word_embeddings = []

    print("Loading embeddings...") 
    start = time.time()
    embeddings = gensim.models.KeyedVectors.load_word2vec_format(embeddings_path, binary=True)
    end = time.time()
    print(f"Completed in {end - start} seconds.")
    embeddings_dim = 300 #len(embeddings.wv[embeddings.vocab.keys()[0]])

    word2Idx["PADDING_TOKEN"] = 0
    vector = np.zeros(embeddings_dim)
    word_embeddings.append(vector)

    word2Idx["UNKNOWN_TOKEN"] = 1
    vector = np.random.uniform(-0.25, 0.25, embeddings_dim)
    word_embeddings.append(vector)


    # loop through each word in embeddings
    for word in embeddings.vocab:
        if word.lower() in words:
            vector = embeddings.wv[word]
            word_embeddings.append(vector)
            word2Idx[word] = len(word2Idx)

    word_embeddings = np.array(word_embeddings)
    print(f"Found embeddings for {word_embeddings.shape[0]} of {len(words)} words.")
    
    return word_embeddings, word2Idx, label2Idx, idx2Label

def text_to_indices(sentences, word2Idx, label2Idx):
    unknown_idx = word2Idx['UNKNOWN_TOKEN']
    padding_idx = word2Idx['PADDING_TOKEN']

    X = []
    Y = []

    null_label = 'O'

    for sentence in sentences:
        word_indices = []
        label_indices = []

        for word, label in sentence:
            if word in word2Idx:
                wordIdx = word2Idx[word]
            elif word.lower() in word2Idx:
                wordIdx = word2Idx[word.lower()]
            else:
                wordIdx = unknown_idx
            word_indices.append(wordIdx)
            label_indices.append(label2Idx[label])

        X.append(word_indices)
        Y.append(label_indices)

    return X, Y

def pad_sentences(X, Y, word2Idx, label2Idx):
    pad_token = word2Idx['PADDING_TOKEN']
    pad_label = label2Idx['O']
    max_sentence_length = 0

    for sentence in X:
        max_sentence_length = max(max_sentence_length, len(sentence))

    print(f"Padding sentences to length {max_sentence_length} with padding token {pad_token}.")

    for i, sentence in enumerate(X):
        while len(sentence) < max_sentence_length:
            X[i].append(pad_token)
            Y[i].append(pad_label)

    return torch.LongTensor(X), torch.LongTensor(Y)
        


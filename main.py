# Reading all the email text files and keeping the ham/spam information in the label variable!
# 1--> spam & 0-->ham

import glob  # using these modules to find all the .txt e-mail files and initialize variables keeping data
import os

emails, labels = [], []

# Appending all SPAM emails text files in a list:-

fp = 'C:/Users/Ndugu/Desktop/Spam E-mail detection/spam'
for fname in glob.glob(os.path.join(fp, '*.txt')):
    f = open(fname, 'r', encoding = "ISO-8859-1")
    emails.append(f.read())
    labels.append(1)

# Appending all HAM emails text files in a list.

fp = 'C:/Users/Ndugu/Desktop/Spam E-mail detection/ham'

for fname in glob.glob(os.path.join(fp, '*.txt')):
    f = open(fname, 'r', encoding = "ISO-8859-1")
    emails.append(f.read())
    labels.append(0)

# NLTK part of cleaning the text data:- 

from nltk.corpus import names
from nltk.stem import WordNetLemmatizer
def letters_only(astr):
    return astr.isalpha()
all_names = set(names.words())
lem = WordNetLemmatizer()

# Storing cleaned e-mails text files:- 

def clean_text(docs):
    cleaned_docs = []

    for doc in docs:
        cleaned_docs.append(' '.join([lem.lemmatize(word.lower()) for word in doc.split() if letters_only(word) and word not in all_names]))
    return cleaned_docs
cleaned_emails = clean_text(emails)

#Removes stop words and extracts features:-

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(stop_words="english", max_features=500)


# The vectorizer will consider 500 most frequent terms. It turns the document matrix into a term doc matrix where each
# row is a term freq sparse vector for a doc and an email.

term_docs = cv.fit_transform(cleaned_emails)

feature_mapping = cv.vocabulary_

# Grouping the data by label:- 

def get_label_index(labels):
    from collections import defaultdict
    label_index = defaultdict(list)
    for index, label in enumerate(labels):
        label_index[label].append(index)
    return label_index
label_index = get_label_index(labels)

# Calculating the prior:- 

def get_prior(label_index):
    prior = {label: len(index) for label, index in label_index.iteritems()}
    total_count = sum(prior.values())
    for label in prior:
        prior[label] /= float(total_count)
    return prior
prior = get_prior(label_index)

# Calculating the likelihood :- 

import numpy as np
def get_likelihood(term_document_matrix, label_index, smoothing=0):
    likelihood = {}
    for label, index in label_index.iteritems():
        likelihood[label] = term_document_matrix[index, :].sum(axis=0) + smoothing
        likelihood[label] = np.asarray(likelihood[label]) [0]
        total_count = likelihood[label].sum()
        likelihood[label] = likelihood[label] / float(total_count)
    return likelihood

smoothing = 1
likelihood = get_likelihood(term_docs, label_index, smoothing)


# Calculating the posterior:- 

def get_posterior(term_document_matrix, prior, likelihood):
    num_docs = term_document_matrix.shape[0]
    posteriors = []

    for i in range(num_docs):
        posterior = {key: np.log(prior_label) for key, prior_label in prior.iteritems()}

    for label, likelihood_label in likelihood.iteritems():
        term_document_vector = term_document_matrix.getrow(i)
        counts = term_document_vector.data
        indices = term_document_vector.indices

        for count, index in zip(counts, indices):
            posterior[label] += np.log(likelihood_label[index]) * count

    min_log_posterior = min(posterior.values())

    for label in posterior:
        try:
            posterior[label] = np.exp(posterior[label] - min_log_posterior)
        except:
            posterior[label] = float('inf')

    sum_posterior = sum(posterior.values())

    for label in posterior:

        if posterior[label] == float('inf'):
            posterior[label] = 1.0

        else:
            posterior[label] /= sum_posterior

    posteriors.append(posterior.copy())

    return posteriors

    
#Testing with an email from another dataset:-
# This list has 2 test mails!

emails_test = [
    '''Subject: flat screens
    hello ,
    please call or contact regarding the other flat screens
    requested .
    trisha tlapek - eb 3132 b
    michael sergeev - eb 3132 a
    also the sun blocker that was taken away from eb 3131 a .
    trisha should two monitors also michael .
    thanks
    kevin moore''',

    '''Subject: having problems in bed ? we can help !
    cialis allows men to enjoy a fully normal sex life without
    having to plan the sexual act .
    if we let things terrify us, life will not be worth living
    brevity is the soul of lingerie .
    suspician always haunts the guilty mind .''',
]

cleaned_test = clean_text(emails_test)
term_docs_test = cv.transform(cleaned_test)
posterior = get_posterior(term_docs_test, prior, likelihood)
print(posterior)
from __future__ import division
import sys
import os.path
import numpy as np

import util

USAGE = "%s <test data folder> <spam folder> <ham folder>"

def get_counts(file_list):
    """
    Computes counts for each word that occurs in the files in file_list.

    Inputs
    ------
    file_list : a list of filenames, suitable for use with open() or 
                util.get_words_in_file()

    Output
    ------
    A dict whose keys are words, and whose values are the number of files the
    key occurred in.
    """
    ### TODO: Comment out the following line and write your code here
    word_dict = util.Counter()
    for file in file_list:
        words = set(util.get_words_in_file(file))
        for item in words:
            word_dict[item] += 1
    return word_dict


def get_log_probabilities(file_list):
    """
    Computes log-probabilities for each word that occurs in the files in 
    file_list.

    Input
    -----
    file_list : a list of filenames, suitable for use with open() or 
                util.get_words_in_file()

    Output
    ------
    A dict whose keys are words, and whose values are the log of the smoothed
    estimate of the fraction of files the key occurred in.

    Hint
    ----
    The data structure util.DefaultDict will be useful to you here, as will the
    get_counts() helper above.
    """
    ### TODO: Comment out the following line and write your code here
    number_of_emails = len(file_list)
    word_count = get_counts(file_list)
    word_prob = util.DefaultDict(lambda: -np.log((number_of_emails + 2)))
    for word in word_count:
        log_count = np.log(word_count[word] + 1) - np.log((number_of_emails + 2))
        word_prob[word] = log_count
    return word_prob

def learn_distributions(file_lists_by_category):
    """
    Input
    -----
    A two-element list. The first element is a list of spam files, 
    and the second element is a list of ham (non-spam) files.

    Output
    ------
    (log_probabilities_by_category, log_prior)

    log_probabilities_by_category : A list whose first element is a smoothed
                                    estimate for log P(y=w_j|c=spam) (as a dict,
                                    just as in get_log_probabilities above), and
                                    whose second element is the same for c=ham.

    log_prior_by_category : A list of estimates for the log-probabilities for
                            each class:
                            [est. for log P(c=spam), est. for log P(c=ham)]
    """
    ### TODO: Comment out the following line and write your code here

    log_probabilities_by_category = [None] * 2
    log_prior_by_category = [None] * 2

    total_emails = len(file_lists_by_category[0]) + len(file_lists_by_category[1])
    s = len(file_lists_by_category[0]) / total_emails - .7
    print(s)
    log_prior_by_category[0] = np.log(s)

    log_prior_by_category[1] = np.log(1 - s)

    log_probabilities_by_category[0] = get_log_probabilities(file_lists_by_category[0])

    log_probabilities_by_category[1] = get_log_probabilities(file_lists_by_category[1])

    return log_probabilities_by_category, log_prior_by_category


def classify_message(message_filename,
                     log_probabilities_by_category,
                     log_prior_by_category,
                     names = ['spam', 'ham']):
    """
    Uses Naive Bayes classification to classify the message in the given file.

    Inputs
    ------
    message_filename : name of the file containing the message to be classified

    log_probabilities_by_category : See output of learn_distributions

    log_prior_by_category : See output of learn_distributions

    names : labels for each class (for this problem set, will always be just 
            spam and ham).

    Output
    ------
    One of the labels in names.
    """
    ### TODO: Comment out the following line and write your code here
    email_dict = set(util.get_words_in_file(message_filename))
    spam_prob = 0
    ham_prob = 0
    words = set(log_probabilities_by_category[0].keys())
    words.update(set(log_probabilities_by_category[1].keys()))
    words = set(words)

    for word in words:
        if word in email_dict:
            spam_prob += log_probabilities_by_category[0][word]
            ham_prob += log_probabilities_by_category[1][word]
        else:
            spam_exp = np.exp(log_probabilities_by_category[0][word])
            spam_prob += np.log(1 - spam_exp)
            ham_exp = np.exp(log_probabilities_by_category[1][word])
            ham_prob += np.log(1 - ham_exp)
    log_spam = log_prior_by_category[0] + spam_prob
    log_ham = log_prior_by_category[1] + ham_prob

    if log_spam - log_ham > 0:
        return names[0]
    else:
        return names[1]

if __name__ == '__main__':
    ### Read arguments
    if len(sys.argv) != 4:
        print(USAGE % sys.argv[0])
    testing_folder = sys.argv[1]
    (spam_folder, ham_folder) = sys.argv[2:4]

    ### Learn the distributions
    file_lists = []
    for folder in (spam_folder, ham_folder):
        file_lists.append(util.get_files_in_folder(folder))
    (log_probabilities_by_category, log_priors_by_category) = \
            learn_distributions(file_lists)

    # Here, columns and rows are indexed by 0 = 'spam' and 1 = 'ham'
    # rows correspond to true label, columns correspond to guessed label
    performance_measures = np.zeros([2,2])

    ### Classify and measure performance
    for filename in (util.get_files_in_folder(testing_folder)):
        ## Classify
        label = classify_message(filename,
                                 log_probabilities_by_category,
                                 log_priors_by_category,
                                 ['spam', 'ham'])
        ## Measure performance
        # Use the filename to determine the true label
        base = os.path.basename(filename)
        true_index = ('ham' in base)
        guessed_index = (label == 'ham')
        performance_measures[int(true_index), int(guessed_index)] += 1


        # Uncomment this line to see which files your classifier
        # gets right/wrong:
        print("%s : %s" %(label, filename))

    template="You correctly classified %d out of %d spam messages, and %d out of %d ham messages."
    # Correct counts are on the diagonal
    correct = np.diag(performance_measures)
    # totals are obtained by summing across guessed labels
    totals = np.sum(performance_measures, 1)
    print(template % (correct[0],
                      totals[0],
                      correct[1],
                      totals[1]))

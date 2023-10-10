# -*- coding: utf-8 -*-
"""
NB sentiment analyser. 

Start code.
"""
import argparse
import pandas
import string
import nltk
import copy
import numpy as np
import matplotlib.pyplot as plt
import itertools
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

snowball = SnowballStemmer("english")
"""
Configuration parameters:
"""
# Apply lowercasing during preprocessing
LOWERCASE = True
# Remove punctuations from tokens (barring exclamation marks)
PUNCTUATION = True
# Stem words during preprocessing
STEMMING = False
# Remove stopwords during preprocessing
STOPWORDS = True
# Perform Laplace smoothing during training
LAPLACE = True

# Classify dev dataset if true, else classify test
IS_DEV_SET = True

"""
IMPORTANT, modify this part with your details
"""
USER_ID = "acb20zc" #your unique student ID, i.e. the IDs starting with "acp", "mm" etc that you use to login into MUSE 

def parse_args():
    parser=argparse.ArgumentParser(description="A Naive Bayes Sentiment Analyser for the Rotten Tomatoes Movie Reviews dataset")
    parser.add_argument("training")
    parser.add_argument("dev")
    parser.add_argument("test")
    parser.add_argument("-classes", type=int)
    parser.add_argument('-features', type=str, default="all_words", choices=["all_words", "features"])
    parser.add_argument('-output_files', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('-confusion_matrix', action=argparse.BooleanOptionalAction, default=False)
    args=parser.parse_args()
    return args


def main():
    
    inputs=parse_args()
    
    #input files
    training = inputs.training
    dev = inputs.dev
    test = inputs.test
    
    #number of classes
    number_classes = inputs.classes
    
    #accepted values "features" to use your features or "all_words" to use all words (default = all_words)
    features = inputs.features
    
    #whether to save the predictions for dev and test on files (default = no files)
    output_files = inputs.output_files
     
    
    #whether to print confusion matrix (default = no confusion matrix)
    confusion_matrix = inputs.confusion_matrix
    
    """
    ADD YOUR CODE HERE
    Create functions and classes, using the best practices of Software Engineering
    """
    preprocessed_train_data = Preprocessor(training, features, number_classes, IS_DEV_SET)
    if IS_DEV_SET:    
        set_to_classify = Preprocessor(dev, features, number_classes, IS_DEV_SET)
    else:
        set_to_classify = Preprocessor(test, features, number_classes, IS_DEV_SET)
    
    trained_data = Train(preprocessed_train_data.df)
    classified_data = Classify(set_to_classify, number_classes, trained_data, features)
    evaluation = Evaluate(classified_data, set_to_classify, number_classes, confusion_matrix, IS_DEV_SET)
    
    if output_files:
        output_to_file(classified_data.classified, IS_DEV_SET, number_classes)
    #You need to change this in order to return your macro-F1 score for the dev set
    if IS_DEV_SET:
        f1_score = evaluation.macro_f1_score
        """
        IMPORTANT: your code should return the lines below. 
        However, make sure you are also implementing a function to save the class predictions on dev and test sets as specified in the assignment handout
        """
        print("Student\tNumber of classes\tFeatures\tmacro-F1(dev)\tAccuracy(dev)")
        print("%s\t%d\t%s\t%f" % (USER_ID, number_classes, features, f1_score))

class Preprocessor():
    """
    Preprocessing class. Used to map 5 sentiment classes to 3 sentiment classes.
    Can perform stemming, lowercasing, expanding contractions, and removing punctuations.
    
    Parameters:
    filename - Name of training set
    features - Feature to use (all_words/features)
    class_num - Which sentiment scale to use (3 or 5)
    """
    def __init__(self, filename, features, class_num, IS_DEV_SET):
        self.filename = filename
        self.features = features
        self.df = pandas.read_csv(self.filename, sep='\t')
        self.class_num = class_num
        self.stoplist = stopwords.words("english")
        if self.class_num == 3 and IS_DEV_SET:
            self.df["Sentiment"] = self.df["Sentiment"].apply(self.three_value_sentiment_converter)
            
        if LOWERCASE:
            self.df["Phrase"] = self.df["Phrase"].str.lower()   
    
        self.df["Phrase"] = self.df["Phrase"].apply(self.tokenize_stem_stoplist_and_remove_punctuation)
    def three_value_sentiment_converter(self, sentiment):
        """
        Maps 5 sentiments to 3 sentiments, if need be.
        
        Parameters:
        sentiment - The value of a sentence's sentiment.
        Returns:
        A sentiment value mapped to the 3 sentiment model.
        """
        
        if sentiment == 0 or sentiment == 1:
            return 0
        elif sentiment == 2:
            return 1
        else:
            return 2
            
    def tokenize_stem_stoplist_and_remove_punctuation(self, phrase):
        """
        Tokenizes a single sentence. Can also perform stemming, stoplist removal, 
        and punctuation removal if set to do so.
        
        Parameters:
        phrase - A sentence from the training set.
        
        Returns:
        tokens or stemmed_tokens. tokens is returned if stemming is not done, 
        otherwise stemmed_tokens is returned.
        
        tokens - A list of tokens taken from phrase. Tokens can have its punctuations
        optionally stripped and stoplist words removed. 
        stemmed_tokens - A list of tokens taken from phrase. Tokens are stemmed.
        Optionally, stemmed_tokens can have its punctuations stripped and stoplist words
        removed.
        """
        tokens = nltk.word_tokenize(phrase)
        stemmed_tokens = []
        if PUNCTUATION or STEMMING or STOPWORDS:
            # Import list of punctuations, 
            # tweak it to ignore exclamation marks and add other punctuations
            punctuations = list(string.punctuation)
            punctuations.remove("!")
            punctuations.append("''")
            punctuations.append(".")
            punctuations.append("...")
            punctuations.append("``")
       
            for token in tokens:
                
                if PUNCTUATION and (token in punctuations):
                    tokens.remove(token)
                    continue
                if STOPWORDS and (token in self.stoplist):
                    tokens.remove(token)
                    continue
                
                if STEMMING:
                    stemmed = snowball.stem(token)
                    stemmed_tokens.append(stemmed)
                    continue

        if STEMMING:
            return stemmed_tokens
        
        else:
            return tokens
        
            
            
            
class Train():
    """
    Trains the model after preprocessing the training set. Able to calculate
    the likelihood of sentiments and words appearing in different sentiments.
    
    Parameters:
    data - Preprocessed training set
    """
    def __init__(self, data):
        self.data = data
        self.senti_word_counts = self.count_senti_and_words()
        self.num_senti = self.senti_word_counts[0]
        self.num_word = self.senti_word_counts[1]
        self.likelihood = self.calculate_likelihood()
        self.priors = self.senti_counts_to_priors(self.num_senti)
        
    def count_senti_and_words(self):
        """
        Outputs the count of sentiments and occurrences of words in different sentiments.
        
        Returns:
        senti_counts - A dictionary with the keys being the sentiment classes' values.
        The values are occurrences of the sentiment classes.
        word_counts - A dictionary of dictionaries. First level keys are words that
        occurred in the training set. Second level keys are the sentiment classes' values.
        The values are the number of words that occurred within each sentiment class.
        """
        data = self.data
        senti_counts = {}
        word_counts = {}
        
        # Iterates through all rows in the dataframe to count occurrences of sentiments
        for index, row in data.iterrows():
            sentiment = row["Sentiment"]
            phrase = row["Phrase"]
            
            if sentiment not in senti_counts:
                senti_counts[sentiment] = 0
            senti_counts[sentiment] += 1
            
            # After counting the senti, move onto counting words
            for word in phrase:
                
                if word not in word_counts:
                    word_counts[word] = {}
                if sentiment not in word_counts[word]:
                    word_counts[word][sentiment] = 0
                word_counts[word][sentiment] += 1
        
        return (senti_counts, word_counts)
    
    def calculate_likelihood(self):
        """
        Calculates likelihoods of words belonging in which sentiments
        
        Returns:
        word_counts - Same structure as the word_counts dictionary in count_senti_and_words.
        Value contains likelihood of a word in each sentiment class instead.
        """
        senti_counts = self.num_senti
        word_counts = copy.deepcopy(self.num_word)
        # Iterates through previous output to calculate likelihood
        for word in word_counts:
            
            for sentiments in word_counts[word]:
                word_sent_num = word_counts[word][sentiments]
                sentiment_size = senti_counts[sentiments]
                
                word_counts[word][sentiments] = word_sent_num/sentiment_size
        return word_counts
    
    def senti_counts_to_priors(self, senti):
        """
        Calculates prior probabilities of sentiments occurring in the dataset.
        
        Parameters:
        senti - senti_counts from count_senti_and_words.
        
        Returns:
        senti - A dictionary with sentiment classes as keys. The value is the
        prior probability of those sentiment classes.
        """
        total_class = 0
        
        # Tally up total # of sentiments in training set
        for senti_class in senti:
            total_class += senti[senti_class]
        
        # Iterate through all sentiment classes to get prior prob.
        for senti_class in senti:
            senti[senti_class] = senti[senti_class] / total_class
            
        return senti
        
class Classify():
    """
    Classifies a dev/test set with a Naive Bayes Classifier after the model has 
    been trained. Can extract features based on a list of opinion words, and
    the classifier can perform Laplace smoothing.
    
    Parameters:
    data - Preprocessed dev/test set
    class_num - Which sentiment scale to use (3 or 5)
    trained_data - A Train object
    features - Feature to use (all_words/features)
    """
    def __init__(self, data, class_num, trained_data, features):
        # A list of ~6800 opinion words by Minqing Hu and Bing Liu.
        emotion_words = pandas.read_csv("words.txt", header=0,  skip_blank_lines=False)
        self.data = data
        self.features = features
        self.class_num = class_num
        self.emotion_list = emotion_words["Words"].tolist()
        
        if self.class_num == 3:
            self.classes = [0,1,2]
        else:
            self.classes = [0,1,2,3,4]
            
        self.trained = trained_data
        self.classified = self.classify_dataset()
        
    
    def classify_sentence(self, phrase):
        """
        Classifies a single sentence by calculating the posterior probability of
        all sentiments. Assigns the sentence the sentiment with highest posterior prob.
        
        Parameters:
        phrase: A sentence from the preprocessed dev/test set.
        
        Return:
        posteriors.index(most_likely): Classification of the sentence. 
        """
        posteriors = []
        
        
        for review_class in self.classes:
            prior_prob = self.trained.priors[review_class]
            for word in phrase:
                # When current word appears in training set
                if word in self.trained.num_word:
                    # When considering all words or using feature selection where the current word is in the emotions list
                    if (self.features != "features") or (self.features == "features" and word in self.emotion_list):
                        # When the current word has a likelihood for the current sentiment class
                        if review_class in self.trained.likelihood[word]:    
                            prior_prob *= self.trained.likelihood[word][review_class]
                        # When the current word doesn't have a likelihood for the current sentiment class
                        else:
                            if LAPLACE:
                                class_total = self.trained.num_senti[review_class]
                                distinct_features = len(self.trained.num_word)
                                prior_prob *= 1/(class_total+distinct_features)
                            else:
                                prior_prob *= 0
                    # When word does not appear in emotions list when using feature selection            
                    else:
                        continue
                    
            posteriors.append(prior_prob)
            
        most_likely = max(posteriors)
        return posteriors.index(most_likely)
            
        
            
    def classify_dataset(self):
        """
        Classifies the entire dev/test dataset by iterating row by row and
        performing classify_sentence on each sentence.

        Return:
        posteriors - A dictionary with a sentence_id key, with sentiment class being the value.
        """
        
        posteriors = {}
        for index, row in self.data.df.iterrows():
            phrase = row["Phrase"]
            sentence_id = row["SentenceId"]
            phrase_senti = self.classify_sentence(phrase)
            posteriors[sentence_id] = phrase_senti
        return posteriors
            
class Evaluate:
    """
    NOTE: DO NOT USE ON TEST SETS! Evaluate only works when given labelled data.
    Evaluates the model's classification through confusion matrices and the macro f1 score.
    
    Parameters:
    classified - A Classify object
    actual - The dataset to compare to
    class_num - Which sentiment scale to use (3 or 5)
    plot_cm - True to plot the confusion matrix, False otherwise
    is_dev - True if using dev set, False if using test set
    """
    def __init__(self, classified, actual, class_num, plot_cm, is_dev):
        self.c_data = classified.classified
        self.a_data = actual.df
        self.class_num = class_num
        if self.class_num == 3:
            self.classes = [0,1,2]
        else:
            self.classes = [0,1,2,3,4]

        if is_dev:
            self.con_mat = self.calc_confusion_matrix()
            (self.macro_f1_score, self.accuracy) = self.calc_macro_f1()
            if plot_cm:
                self.plot_confusion_matrix(self.con_mat, self.classes)
        
            
    def calc_confusion_matrix(self):
        """
        Calculates the confusion matrix from the classification's results.
        
        Returns:
        con_mat - A 2d numpy array. X-axis = predicted labels, Y-axis = actual labels
        """
        con_mat = np.zeros((self.class_num, self.class_num))
        for index, row in self.a_data.iterrows():
            sentence_id = row["SentenceId"]
            c_senti = self.c_data[sentence_id]
            a_senti = row["Sentiment"]
            con_mat[a_senti][c_senti] += 1
        return con_mat
    
    def calc_macro_f1(self):
        """
        Calculates the macro f1 scores from the classification's results.
        
        Returns:
        macro_f1_score - The macro f1 score.
        accuracy - How accurate the classifier was able to classify sentiments
        """
        if self.class_num == 3:
            classes_f1 = {0: {"tp": 0, "fn": 0, "fp": 0},
                          1: {"tp": 0, "fn": 0, "fp": 0},
                          2: {"tp": 0, "fn": 0, "fp": 0}}
            macro_f1_arr = np.zeros(3)
        else:
            classes_f1 = {0: {"tp": 0, "fn": 0, "fp": 0},
                          1: {"tp": 0, "fn": 0, "fp": 0},
                          2: {"tp": 0, "fn": 0, "fp": 0},
                          3: {"tp": 0, "fn": 0, "fp": 0},
                          4: {"tp": 0, "fn": 0, "fp": 0}}
            macro_f1_arr = np.zeros(5)
            
        # Figure out the number of true positives, false negatives + positives
        # for all sentiment classes
        for predict_class in self.classes:
            for actual_class in self.classes:
                curr_square = self.con_mat[predict_class][actual_class]
                if (predict_class == actual_class):
                    classes_f1[predict_class]["tp"] += curr_square
                else:
                    classes_f1[predict_class]["fp"] += curr_square
                    classes_f1[actual_class]["fn"] += curr_square
        
        # Then, calculate the f1 scores for all sentiment classes
        for senti in classes_f1:
            tp_senti = 0
            fp_senti = 0
            fn_senti = 0
            for measures in classes_f1[senti]:
                if measures == "tp":
                    tp_senti += classes_f1[senti][measures]
                elif measures == "fp":
                    fp_senti += classes_f1[senti][measures]
                elif measures == "fn":
                    fn_senti += classes_f1[senti][measures]
            macro_numer = 2*tp_senti
            macro_demom = 2*tp_senti+fp_senti+fn_senti
            macro_score = macro_numer/macro_demom
            macro_f1_arr[senti] = macro_score
        
        macro_f1_score = np.sum(macro_f1_arr)/self.class_num
        accuracy = np.trace(self.con_mat) / float(np.sum(self.con_mat))
        return (macro_f1_score, accuracy)
        
    def plot_confusion_matrix(self, cm, target_names, t='Confusion matrix') -> None:
        """
        Plots a confusion matrix based on a 2d numpy array. Code is adapted 
        from the SA_PMI_Gradable lab.
        """
        accuracy = np.trace(cm) / float(np.sum(cm))
        misclass = 1 - accuracy

        cmap = plt.get_cmap('Blues')

        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(t)
        plt.colorbar()

        if target_names is not None:
            tick_marks = np.arange(len(target_names))
            plt.xticks(tick_marks, target_names, rotation=45)
            plt.yticks(tick_marks, target_names)

        thresh = cm.max() / 2
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.grid(False)
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
        plt.show()
        
def output_to_file(classified, is_dev, num_class):
    """
    Writes the classification results into a .tsv file.
    
    Parameters:
    classified: A Classify object
    is_dev: True if using dev set, false if using test set
    num_class: Which sentiment scale to use (3 or 5)
    """
    if is_dev:
        set_classify = "dev"
    else:
        set_classify = "test"
    
    if num_class == 3:
        number = "3classes"
    else:
        number = "5classes"
        
    file = "./" + set_classify + "_predictions_" + number + "_acb20zc.tsv"
    with open(file, "wt") as f_output:
        f_output.write("SentenceId" + "\t" + "Sentiment" + "\n")
        for sentence_id in classified:
            f_output.write(str(sentence_id) + "\t" + str(classified[sentence_id]) + "\n")
        
                
                
            
                    
    
        
if __name__ == "__main__":
    main()
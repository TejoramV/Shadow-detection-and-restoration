import csv 
import cv2
import collections
from nltk.metrics import f_measure
from nltk.metrics import precision
from nltk.metrics.agreement import AnnotationTask
from sklearn.metrics import cohen_kappa_score
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews
 
def word_feats(words):
    return dict([(word, True) for word in words])
 
negids = movie_reviews.fileids('neg')
posids = movie_reviews.fileids('pos')
 
negfeats = [(word_feats(movie_reviews.words(fileids=[f])), 'neg') for f in negids]
posfeats = [(word_feats(movie_reviews.words(fileids=[f])), 'pos') for f in posids]
 
negcutoff = len(negfeats)*3/4
poscutoff = len(posfeats)*3/4
 
trainfeats = negfeats[:negcutoff] + posfeats[:poscutoff]
testfeats = negfeats[negcutoff:] + posfeats[poscutoff:]

classifier = NaiveBayesClassifier.train(trainfeats)
refsets = collections.defaultdict(set)
testsets = collections.defaultdict(set)
 
for i, (feats, label) in enumerate(testfeats):
    refsets[label].add(i)
    observed = classifier.classify(feats)
    testsets[observed].add(i)
 
print ('pos precision:', nltk.metrics.precision(refsets['pos'], testsets['pos']))
print ('pos recall:', nltk.metrics.recall(refsets['pos'], testsets['pos']))
print ('pos F-measure:', nltk.metrics.f_measure(refsets['pos'], testsets['pos']))
print ('neg precision:', nltk.metrics.precision(refsets['neg'], testsets['neg']))
print ('neg recall:', nltk.metrics.recall(refsets['neg'], testsets['neg']))
print ('neg F-measure:', nltk.metrics.f_measure(refsets['neg'], testsets['neg']))


fields = [] 
rows = [] 

filename = "../Images/statsbuilding.csv"

with open(filename, 'r') as csvfile: 

    csvreader = csv.reader(csvfile) 
 
    fields = next(csvreader) 
 
    for row in csvreader: 
        rows.append(row) 
    

org=set(rows[1])
orgi=set(rows[3])
orgii=rows[1]
orgiii=rows[3]

print("kappa:",cohen_kappa_score(orgii,orgiii))
print ('Accuracy:', precision(org,orgi)*100)
print ('F-measure:',f_measure(org,orgi)*100)

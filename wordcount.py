# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 16:46:54 2017

@author: Satya
"""
from glob import glob
#from collections import Counter
#import re
#import pickle
import os
from collections import Counter
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.svm import NuSVC



def create_dictionary(path):
    filepaths=glob(os.path.join(path,'*.txt'))
    all_words = []
    for fp in filepaths:
        with open(fp) as f:
            for line in f:
                words = line.split()
                all_words += words
    dictionary = Counter(all_words)
    
    all_keys = dictionary.keys()
    for key in all_keys:
        if key.isalpha() == False:
            del dictionary[key]
        if len(key)==1:
            del dictionary[key]
    dictionary=dictionary.most_common(3000)          
    return dictionary

def feature_extraction(path):
    filepaths = glob(os.path.join(path,'*.txt')) 
    feature_matrix = np.zeros([len(filepaths),3000])
    for docId,fp in enumerate(filepaths):
        file_words=[]
        with open(fp) as f:
            for line in f:
                words = line.split()
                file_words += words
            
            file_dict=Counter(file_words)
            
            for item in file_dict:
                for wordId,key in enumerate(dictionary):                
                    if key[0] == item:
                        feature_matrix[docId,wordId] =  file_dict[item]   
    
    return feature_matrix
    
        
train_path='C:\\Users\\Satya\\Desktop\\NLP\\train-mails'

dictionary=create_dictionary(train_path)

training_feature_matrix=feature_extraction(train_path)

train_label = np.zeros(702)
train_label[351:702] = 1
           
svm_model = LinearSVC()
svm_model.fit(training_feature_matrix,train_label)     

gaussiannb = GaussianNB()
gaussiannb.fit(training_feature_matrix,train_label)

multinomial_nb = MultinomialNB()
multinomial_nb.fit(training_feature_matrix,train_label)

bernoulli_nb = BernoulliNB()
bernoulli_nb.fit(training_feature_matrix,train_label)

nu_svc = NuSVC()
nu_svc.fit(training_feature_matrix,train_label)

svc = SVC()
svc.fit(training_feature_matrix,train_label)

test_path = 'C:\\Users\\Satya\\Desktop\\NLP\\test-mails'  
test_feature_matrix=feature_extraction(test_path)
test_label = np.zeros(260)
test_label[130:260] = 1
   

predicted_label1 = svm_model.predict(test_feature_matrix)       
predicted_label2 = gaussiannb.predict(test_feature_matrix)         
predicted_label3 = multinomial_nb.predict(test_feature_matrix)         
predicted_label4 = bernoulli_nb.predict(test_feature_matrix)
predicted_label5 = svc.predict(test_feature_matrix)
predicted_label6 = nu_svc.predict(test_feature_matrix)


print sum(test_label == predicted_label1)
print sum(test_label == predicted_label2)
print sum(test_label == predicted_label3)
print sum(test_label == predicted_label4)
print sum(test_label == predicted_label5)
print sum(test_label == predicted_label6)

print confusion_matrix(test_label,predicted_label1)
print confusion_matrix(test_label,predicted_label2)
print confusion_matrix(test_label,predicted_label3)
print confusion_matrix(test_label,predicted_label4)
print confusion_matrix(test_label,predicted_label5)
print confusion_matrix(test_label,predicted_label6)


#print dictionary
#np.save('dictionary.npy',dictionary)
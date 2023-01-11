# import libraries
import numpy as np
import os
from sklearn.svm import LinearSVC, SVC, NuSVC
from collections import Counter
from sklearn.metrics import accuracy_score, confusion_matrix




# Helper functions to create dictionary and extract features from the corpus for model development
def make_Dictionary(train_dir):
    emails = [os.path.join(train_dir,f) for f in os.listdir(train_dir)]    
    all_words = []       
    for mail in emails:    
        with open(mail) as m:
            for i,line in enumerate(m):
                if i == 2:  #Body of email is only 3rd line of text file
                    words = line.split()
                    all_words += words
    
    dictionary = Counter(all_words)
    # Code for non-word removal
   #list_to_remove = dictionary.keys()
    list_to_remove = list(dictionary.keys()) 
    for item in list_to_remove:
        if item.isalpha() == False: 
            del dictionary[item]
        elif len(item) == 1:
            del dictionary[item]
    dictionary = dictionary.most_common(3000) 
    return dictionary

def extract_features(mail_dir): 
    files = [os.path.join(mail_dir,fi) for fi in os.listdir(mail_dir)]
    features_matrix = np.zeros((len(files),3000))
    docID = 0;
    for fil in files:
      with open(fil) as fi:
        for i,line in enumerate(fi):
          if i == 2:
            words = line.split()
            for word in words:
              wordID = 0
              for i,d in enumerate(dictionary):
                if d[0] == word:
                  wordID = i
                  features_matrix[docID,wordID] = words.count(word)
        docID = docID + 1     
    return features_matrix

# Use the above two functions to load dataset, create training, test splits for model development

train_dir = 'C:/Users/Ruben/Downloads/CSE 5120 emailClassification/ling-spam/train-mails'
dictionary = make_Dictionary(train_dir)

# Prepare feature vectors per training email and its labels

train_labels = np.zeros(702)
train_labels[351:701] = 1
train_matrix = extract_features(train_dir)


# Train SVM model

model = LinearSVC()
model.fit(train_matrix,train_labels)



# Test SVM model on unseen emails (test_matrix created from extract_features function)

test_dir = 'C:/Users/Ruben/Downloads/CSE 5120 emailClassification/ling-spam/test-mails'
test_matrix = extract_features(test_dir)
test_labels = np.zeros(260)
test_labels[130:260] = 1
result = model.predict(test_matrix)
print (confusion_matrix(test_labels,result))
print (accuracy_score(test_labels, result))


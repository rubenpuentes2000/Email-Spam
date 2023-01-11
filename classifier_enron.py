# import libraries
import numpy as np
import os
from sklearn.svm import LinearSVC, SVC, NuSVC
from collections import Counter
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import pickle


# f=open(filename,wb) # both strings 
# pickle.dump(confusion_matrix,f)
# f.close()



# =============================================================================
# C:/Users/Ruben/Downloads/CSE 5120 emailClassification/enron-spam 
# =============================================================================

# Helper functions to create dictionary and extract features from the corpus for model development
def make_Dictionary(train_dir):
    emails = [os.path.join(train_dir,f) for f in os.listdir(train_dir)]    
    all_words = []       
    for mail in emails:    
        with open(mail,'r',encoding='utf-8',errors='ignore') as m:
            for i,line in enumerate(m):
                if i == 2:  #Body of email is only 3rd line of text file
                    words = line.split()
                    all_words += words
    
    dictionary = Counter(all_words)
    # Code for non-word removal
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
    train_labels = np.zeros(33716)
    for fil in files:
      with open(fil,'r',encoding='utf-8',errors='ignore') as fi:
        for i,line in enumerate(fi):
          if i == 2:
            words = line.split()
            for word in words:
              wordID = 0
              for i,d in enumerate(dictionary):
                if d[0] == word:
                  wordID = i
                  features_matrix[docID,wordID] = words.count(word)
        train_labels[docID] = int(fil.split(".")[-2]== 'spam')
        docID = docID + 1     
    return features_matrix, train_labels

# Use the above two functions to load dataset, create training, test splits for model development

train_dir = 'C:/Users/Ruben/Downloads/CSE 5120 emailClassification/enron-spam/enron-spam'
dictionary = make_Dictionary(train_dir)

# Train SVM model

#train_labels = np.zeros(33716)
#train_labels[16858:33716] = 1

train_matrix, train_labels = extract_features(train_dir)



# training_data, testing_data = train_test_split 

x_train, x_test, y_train, y_test = train_test_split(train_matrix, train_labels, test_size=0.4, random_state=0)      

model = LinearSVC()
model.fit(x_train,y_train)                          


# Test SVM model on unseen emails (test_matrix created from extract_features function)

#test_dir = 'C:/Users/Ruben/Downloads/CSE 5120 emailClassification/enron-spam/enron-spam            '
#test_matrix = extract_features(test_dir)
#test_labels = np.zeros(260)


result = model.predict(x_test)

print (confusion_matrix(y_test,result))
print (accuracy_score(y_test, result))

# Save your model as .sav file to upload with your submission

p_output = open("emailClassifer_eron.txt","wb")
pickle.dump(result, p_output)
p_output.close()



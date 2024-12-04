import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from AudioFeatureExtractor import AudioFeatureExtractor
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import KFold
import os
#from preprocess_data import preprocess_data, create_splits

# Feature extraction.  
#Number of segments to split signal
nsegments=1
#number of cepstral coefficients
num_mfcc=10
# Number of samples to use for testing
t_size = 0.2
audioFeatures = None
if not "MasterAudio.csv" in os.listdir("./data"):
    for folder in os.listdir("./data"):
        
        directory = os.path.join("./data", folder)
        if os.path.isdir(directory) and not directory == "./data/.DS_Store":
            audioProcessor = AudioFeatureExtractor(directory)
            audioFeaturesLine = audioProcessor.constructMFCCFeatures(nsegments, num_mfcc)
            if audioFeatures is None:
                audioFeatures = audioFeaturesLine
            else:
                audioFeatures = pd.concat([audioFeatures, audioFeaturesLine], axis=0)
            print(directory)
        else: 
            continue
    audioFeatures.to_csv("MasterAudio.csv")
else: 
    audioFeatures = pd.read_csv("MasterAudio.csv")
print(np.unique(audioFeatures["Target"]))
print(audioFeatures)

#Split data into testing and training using stratified sampling
X = audioFeatures.iloc[:,1:]
Y =audioFeatures["Target"]
#X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=t_size, random_state=1, stratify=Y)
kf = KFold(n_splits=5, shuffle=True, random_state=42)



bestAcc = 0
bestX_test = None
bestY_test = None
bestModel = None
accurs = []

    
    
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
    #Build Classifier
    knn = KNeighborsClassifier(n_neighbors=10, weights='distance',)
    knn.fit(X_train, Y_train)
    #Test perfornmance
    
    #accuracy
    training_accuracy = knn.score(X_train, Y_train)
    test_accuracy = knn.score(X_test, Y_test)
    print("Training Accuracy:", training_accuracy)
    print("Test Accuracy:", test_accuracy)
    accurs.append(test_accuracy)
    if test_accuracy > bestAcc:
        bestAcc = test_accuracy
        bestX_test = X_test
        bestY_test = Y_test
        bestModel = knn
    print("The best Accuracy is", bestAcc)
    
    #Confusion matrix
    my_labels=['0', '1', '2', '3','4','5','6','7','8','9']
    Y_pred = knn.predict(X_test)
    
# Plot non-normalized confusion matrix
titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp = ConfusionMatrixDisplay.from_estimator(bestModel, bestX_test, bestY_test,
                                  display_labels=my_labels,
                                  cmap=plt.cm.Blues,
                                  normalize=normalize)
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)
    
plt.show()
    
    

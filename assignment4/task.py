import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
easy_ham_file = "./easy_ham"
hard_ham_file = "./hard_ham"
spam_file = "./spam"

def read_emails(dir):
    email = []
    for filename in os.listdir(dir):
        with open(os.path.join(dir, filename), "r", encoding='latin-1' ) as file:
            email_text = file.read()
            email.append(email_text)
    return email

EasyHamEmail = read_emails(easy_ham_file)
# HardHamEmail = read_emails(hard_ham_file)
SpamEmail = read_emails(spam_file)


# label the ham_email and spam email, 0-> ham email, 1-> spam email 
ham_email_label = [0] * len(EasyHamEmail)
spam_email_label = [1] * len(SpamEmail)
EasyHam_Train, EasyHam_Test, EasyHam_Labels_Train, EasyHam_Labels_Test = train_test_split(EasyHamEmail, ham_email_label, test_size=0.2, random_state=42)
Spam_Train, Spam_Test,  Spam_Lable_Train, Spam_Lable_Test = train_test_split(SpamEmail, spam_email_label, test_size=0.2, random_state=42) 

X_Train = EasyHam_Train + Spam_Train
Y_Train = EasyHam_Labels_Train + Spam_Lable_Train
X_Test = EasyHam_Test + Spam_Test
Y_Test = EasyHam_Labels_Test + Spam_Lable_Test

# Convert text into numerical features.
vectorizer = CountVectorizer()
X_Train_vector = vectorizer.fit_transform(X_Train)
# print(vectorizer.get_feature_names_out())
X_Test_vector = vectorizer.transform(X_Test)

#  Train naive bayers 
Multinomial_nb = MultinomialNB()
Multinomial_nb.fit(X_Train_vector, Y_Train)
Multinomial_nb_pred = Multinomial_nb.predict(X_Test_vector)
Multinomial_nb_confusionMatrix = confusion_matrix(Y_Test, Multinomial_nb_pred)
# print(Multinomial_nb_confusionMatrix)
multinomial_true_positive_rate = Multinomial_nb_confusionMatrix[1, 1] / sum(Multinomial_nb_confusionMatrix[1, :])
multinomial_false_negative_rate = Multinomial_nb_confusionMatrix[1, 0] / sum(Multinomial_nb_confusionMatrix[1, :])


ConfusionMatrixDisplay(confusion_matrix=Multinomial_nb_confusionMatrix).plot()
plt.show()
#nernoulli 
Bernoulli_nb = BernoulliNB()
Bernoulli_nb.fit(X_Train_vector, Y_Train)
Bernoulli_nb_Pred = Bernoulli_nb.predict(X_Test_vector)
Bernoulli_nb_confusionMatrix = confusion_matrix(Y_Test, Bernoulli_nb_Pred)
ConfusionMatrixDisplay(confusion_matrix=Bernoulli_nb_confusionMatrix).plot()
# Calculate True Positive and False Negative rates for Bernoulli Naive Bayes
bernoulli_true_positive_rate = Bernoulli_nb_confusionMatrix[1, 1] / sum(Bernoulli_nb_confusionMatrix[1, :])
bernoulli_false_negative_rate = Bernoulli_nb_confusionMatrix[1, 0] / sum(Bernoulli_nb_confusionMatrix[1, :])
plt.show()


print("Multinomial Naive Bayes:")
print("True Positive Rate:", multinomial_true_positive_rate)
print("False Negative Rate:", multinomial_false_negative_rate)
print("")

print("Bernoulli Naive Bayes:")
print("True Positive Rate:", bernoulli_true_positive_rate)
print("False Negative Rate:", bernoulli_false_negative_rate)
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score

df = pd.read_excel('10 Comment Unsorted Scenarios2.xlsx')

X = df.drop(['Label', 'Participants', 'Participant no','Feature 1',	'Feature 2'	,'Feature 3',	'Feature 4',	'Feature 5'	,'Feature 6'	,'Feature 7',	'Feature 8',	'Feature 9',	'Feature 10'], axis=1).values
y = df['Label'].values
best_accuracy = 0
best_state = None


results = []

test_score_avg=0
for random_state in range(1, 201):  
  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=random_state)

    pipeline = make_pipeline(
        StandardScaler(),
        SVC(kernel='rbf', C=1)
    )

    scores = cross_val_score(pipeline, X_train, y_train, cv=10)
    cv_accuracy = scores.mean()

    pipeline.fit(X_train, y_train)

    test_score = pipeline.score(X_test, y_test)

    y_predicted = pipeline.predict(X_test)
    f1 = f1_score(y_test, y_predicted)

    results.append((random_state, test_score, cv_accuracy, f1))

    if test_score > best_accuracy:
        best_accuracy = test_score
        best_state = random_state

    print(f'Random State: {random_state} | Test Accuracy: {test_score:.3f} | CV Accuracy: {cv_accuracy:.3f} | F1 Score: {f1:.3f}')
    test_score_avg+=test_score

print(f'\nBest Random State: {best_state} with Test Accuracy: {best_accuracy:.3f}')

print("test score average : " , test_score_avg/200)

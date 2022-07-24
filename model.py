import pandas as pd
import pickle

df = pd.read_csv('hiring_model.csv')
df

df.experience.fillna(0, inplace=True)
df.test_score.fillna(df.test_score.mean(), inplace=True)

def convert_to_int(word):
    word_dict = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5,'six':6,
                'seven':7,'eight':8,'nine':9,'ten':10, 'eleven':11,
                 'zero':0, 0:0}
    return word_dict[word]

X = df.iloc[:, :3]
X.experience = X.experience.apply(lambda x : convert_to_int(x))

y = df.iloc[:,-1]

# Implement linear regression model..

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()

lin_reg.fit(X,y)  # 


#saving model to disk..

pickle.dump(lin_reg, open('model.pkl', 'wb'))
model = pickle.load(open('model.pkl', 'rb'))

print(model.predict([[2,9,6]]))

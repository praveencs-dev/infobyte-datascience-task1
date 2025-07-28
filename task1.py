from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
iris=load_iris()
x=iris.data
y=iris.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=DecisionTreeClassifier()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
accuracy= accuracy_score(y_test,y_pred)
print("model accury on test data:",accuracy)
print("predict your own flower/n")
sepal_lenght=float(input("enter sepal lenght(cm):"))
sepal_width=float(input("enter sepal width(cm):"))
petal_lenght=float(input("enter petal lenght(cm):"))
petal_lenght=float(input("enter petal lenght(cm):"))
flower=[[sepal_lenght,sepal_width,petal_lenght,petal_lenght]]
prediction=model.predict(flower)[0]
flower_name=iris.target_names[prediction]
print("the flower name is:",flower_name.title())

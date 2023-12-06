from django.shortcuts import render , redirect 
import pickle 
import pandas as pd 


df = pd.read_csv("covid_toy.csv")
print(df.head())

df = df.dropna() 

from sklearn.preprocessing import LabelEncoder 
lb = LabelEncoder() 
df['gender'] = lb.fit_transform(df['gender'])
df['cough'] = lb.fit_transform(df['cough'])
df['city'] = lb.fit_transform(df['city'])
df['has_covid'] = lb.fit_transform(df['has_covid'])

x = df.drop(columns = ['has_covid'] ,axis = 1) 
y = df['has_covid'] 



print(x) 
print(df.isnull().sum()) 


from sklearn.model_selection import train_test_split
x_train, x_test, y_train,y_test = train_test_split(x,y,test_size = 0.2)


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)

knn_ans = knn.fit(x_train, y_train)




# Create your views here.
def home(request):
    return render(request , "index.html") 



def predict(request):
    if request.method == 'POST':
        a = request.POST.get('age') 
        age = int(a) 
        s = request.POST.get('gender') 
        gender = int(s) 
        t = request.POST.get('fever') 
        fever = float(t) 
        u = request.POST.get('cough') 
        cough = int(u) 
        p = request.POST.get('city') 
        city = int(p) 
        
        result = knn_ans.predict([[age , gender , fever , cough , city]])[0] 

        print(result , '>>>>>>>>') 

        if result == 1:
            return render(request , 'index.html', {'label':1})
        else:
            return render(request , 'index.html' , {'label':-1}) 
    else:
        return redirect('/')   

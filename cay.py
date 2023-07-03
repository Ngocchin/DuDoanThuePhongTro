
import tkinter as tk
from tkinter import *
from tkinter import messagebox
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv("BTL_MauHL1.csv")
# data
KC={'N1':0, 'N3':1, 'N5':2, 'N7':3,'N10':4}
Dientich={'N15':0, 'N20':1, 'N25':2, 'N30':3,'N35':4}
thongthoang = {'Y':0, 'N':1}
khepkin={'Y':0, 'N':1}
Giatien={'N1.5':0, 'N2.0':1, 'N2.5':2, 'N3.0':3,'N3.5':4}

# Thue={'Y':0, 'N':1}
data['KC']=data['KC'].map(KC)
data['Dientich']=data['Dientich'].map(Dientich)
data['thongthoang']=data['thongthoang'].map(thongthoang)
data['khepkin']=data['khepkin'].map(khepkin)
data['Giatien']=data['Giatien'].map(Giatien)
# data['Thue']=data['Thue'].map(Thue)

print(data.info())

X = data[['KC','Dientich','thongthoang','khepkin','Giatien']].values
print(X[0:10])

Y = data['Thue']
print(Y[0:10])

from sklearn.model_selection import train_test_split
X_trainset, X_testset, Y_trainset, Y_testset = train_test_splittrain_X, test_X, train_y, test_y= train_test_split(X,Y, test_size=0.5, random_state=0)

SpeciesTree = DecisionTreeClassifier(criterion = 'entropy', max_depth = 4)
SpeciesTree

# X_trainset=data[cotdt]
SpeciesTree.fit(X_trainset, Y_trainset)

predTree = SpeciesTree.predict(X_testset)

# print(predTree [0:5])
# print(Y_testset[0:5])

#Độ chính xác
from sklearn import metrics
import matplotlib.pyplot as plt
print("DecisionTrees's Accuracy: ",metrics.accuracy_score(Y_testset, predTree))

import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
X = data[['KC','Dientich','thongthoang','khepkin','Giatien']].values
fn=data.columns[1:6]
cn = data["Thue"].unique().tolist()
SpeciesTree.fit(X, Y)
fig, axes = plt.subplots(nrows = 1,ncols = 1, figsize = (10,10), dpi = 300)

tree.plot_tree(SpeciesTree,feature_names = fn, class_names = cn, filled = True);
_=tree.plot_tree(SpeciesTree)
# fig.savefig('BTL/cay.jpg')
# print('Done')

# Xay dung cay
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix

my_tree = DecisionTreeClassifier(splitter='random')
my_tree.fit(X_testset, Y_testset)

# Du doan tren du lieu test

# y_pred = my_tree.predict(X_testset)

# cm = confusion_matrix(Y_testset, y_pred)

# print(cm)

# plot_confusion_matrix(my_tree, X_testset, Y_testset)

da = pd.read_csv("Tap_Test.csv")
print(da.head())

KC={'N1':0, 'N3':1, 'N5':2, 'N7':3,'N10':4}
Dientich={'N15':0, 'N20':1, 'N25':2, 'N30':3,'N35':4}
thongthoang = {'Y':0, 'N':1}
khepkin={'Y':0, 'N':1}
Giatien={'N1.5':0, 'N2.0':1, 'N2.5':2, 'N3.0':3,'N3.5':4}

# Thue={'Y':0, 'N':1}
da['KC']=da['KC'].map(KC)
da['Dientich']=da['Dientich'].map(Dientich)
da['thongthoang']=da['thongthoang'].map(thongthoang)
da['khepkin']=da['khepkin'].map(khepkin)
da['Giatien']=da['Giatien'].map(Giatien)
# data['Thue']=data['Thue'].map(Thue)

x = da[['KC','Dientich','thongthoang','khepkin','Giatien','Thue']].values
print(x[15:20])

from sklearn.tree import DecisionTreeRegressor
# X_trainset=data[cotdt]
my_tree.fit(X_testset, Y_testset)

from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
X = da[['KC','Dientich','thongthoang','khepkin','Giatien']].values
fn=da.columns[0:5]
cn = da["Thue"].unique().tolist()
y = da['Thue']
my_tree.fit(X,y)
fig, axes = plt.subplots(nrows = 1,ncols = 1, figsize = (10,10), dpi = 300)

tree.plot_tree(my_tree,feature_names = fn, class_names = cn, filled = True);
_=tree.plot_tree(my_tree)
# fig.savefig('BTL/cay2.jpg')
# print('Done')

#giao diện
master=tk.Tk()
master.title("Dự đoán thuê hay không")
tk.Label(master, text="Nhập vào các thông tin").grid (column=0, row=0)
tk.Label(master, text="Nhập vào khoảng cách").grid (column=0, row=1)
s1 = Entry(master, width=40)
s1.grid(column=1, row=1)
tk.Label(master, text="Nhập vào diện tích").grid (column=0, row=2)
s2 = Entry(master, width=40)
s2.grid(column=1, row=2)
tk.Label(master, text="Thông thoáng").grid (column=0, row=3)
s3 = Entry(master, width=40)
s3.grid(column=1, row=3)
tk.Label(master, text="khép kín").grid (column=0, row=4)
s4 = Entry(master, width=40)
s4.grid(column=1, row=4)
tk.Label(master, text="Nhập Giá tiền").grid (column=0, row=5)
s5 = Entry(master, width=40)
s5.grid(column=1, row=5)

tk.Button(master, text='Exit', command=master.quit).grid(row=6, column=0, sticky=tk.W, pady=6)
def predict():
    bien1=float(s1.get())
    bien2=float(s2.get())
    bien3=float(s3.get())
    bien4=float(s4.get())
    bien5=float(s5.get())
    X_new = [[bien1,bien2,bien3,bien4,bien5]]
    predTree = SpeciesTree.predict(X_new)
    
    # print(predTree)
    messagebox.showinfo("Dự đoán: ", predTree)
tk.Button(master, text='Show Prediction', command=predict).grid(row=6, column=1, sticky=tk.W, pady=6)
master.mainloop()

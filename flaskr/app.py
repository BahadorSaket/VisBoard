import csv
import json
from numpy import genfromtxt
from numpy import matrix
from numpy import linalg
import numpy as np
from scipy.spatial import distance
from flask import Flask, render_template, redirect, url_for,request
from flask import make_response

app = Flask(__name__)
 
@app.route("/")
def home():
    return "hi"
@app.route("/index")
	
@app.route('/login', methods=['GET', 'POST'])
def login():
    message = None
    if request.method == 'POST':
        matrixfromjs = request.form['mydata']
        checker=request.form['counter']
        print(checker)
        np.set_printoptions(precision=1)
        matrix = np.asmatrix(genfromtxt('car.csv', delimiter=','))
        matrix = matrix.T

        #print(np.nan_to_num(matrix))

        mean_x = np.mean(matrix[0,:])
        mean_y = np.mean(matrix[1,:])
        mean_z = np.mean(matrix[2,:])
        mean_vector = np.array([[mean_x],[mean_y],[mean_z]])

        #print('Mean Vector:\n', mean_vector)

        scatter_matrix = np.zeros((3,3))
        for i in range(matrix.shape[1]):
            scatter_matrix += (matrix[:,i].reshape(3,1) - mean_vector).dot(
                (matrix[:,i].reshape(3,1) - mean_vector).T)
				
            #print('Scatter Matrix:\n', scatter_matrix)

        eig_val_sc, eig_vec_sc = np.linalg.eig(scatter_matrix)

        for i in range(len(eig_val_sc)):
            eigvec_sc = eig_vec_sc[:,i].reshape(1,3).T
			
        eig_pairs = [(np.abs(eig_val_sc[i]), eig_vec_sc[:,i])
                for i in range(len(eig_val_sc))]
				
        if checker=='0':
            result= matrix.tolist()
            result=json.dumps(result) 	
        elif checker=='1':
            eig_pairs.sort()
            eig_pairs.reverse()
          #  for i in eig_pairs:
          #      print(i[0],"-",i[1])
            matrix_w = np.hstack((eig_pairs[0][1].reshape(3,1),
                        eig_pairs[1][1].reshape(3,1)))
          #  print('Matrix W:\n', matrix_w)
            transformed_matrix = matrix_w.T.dot(matrix)
          #  print('Transformed Matrix:\n', transformed_matrix)
            result= transformed_matrix.T
            result= result.tolist()
            result=json.dumps(result) 
          #  print(result)
        else:
         #   print(matrixfromjs)
            transformed_matrix = matrixfromjs 
         #   print(transformed_matrix)	
            transformed_matrix = json.loads(transformed_matrix)			
            transformed_matrix =  np.asmatrix(transformed_matrix)
            transformed_matrix = transformed_matrix.T
         #  print('Matrix y:\n', transformed_matrix)
            inverse= matrix.I
            weight = transformed_matrix.dot(inverse)
            first_Eighen = weight[0,:]
            second_Eighen = weight[1,:]
          #  print('first eighen:\n',first_Eighen)
          #  print('second eighen:\n',second_Eighen)
            dis=[]
            counter= 1
            for i in eig_pairs:
                dist1 = distance.cosine(first_Eighen,i[1])
                dist2 = distance.cosine(second_Eighen,i[1])
                dis.append([(dist1+dist2)/2,counter])
                counter=counter+1
            dis = np.sort(dis)  
            result= dis.tolist()
            result=json.dumps(result) 	  

        resp = make_response(result)
        resp.headers['Content-Type'] = "application/json"
        return resp
    return render_template('login.html', message='')
if __name__ == "__main__":
    app.run(debug = True)

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
			 
        mean_attribute=[]
        mean_vector=[]
        for i in range(0,len(matrix)):
            mean_attribute.append(np.mean(matrix[i,:]))
            mean_vector.append([mean_attribute[i]]);
       
        scatter_matrix = np.zeros((len(matrix),len(matrix)))
        for i in range(matrix.shape[1]):
            scatter_matrix += (matrix[:,i].reshape(len(matrix),1) - mean_vector).dot(
                (matrix[:,i].reshape(len(matrix),1) - mean_vector).T)
				
        eig_val_sc, eig_vec_sc = np.linalg.eig(scatter_matrix)

        for i in range(len(eig_val_sc)):
            eigvec_sc = eig_vec_sc[:,i].reshape(1,len(matrix)).T
			
        eig_pairs = [(np.abs(eig_val_sc[i]), eig_vec_sc[:,i])
                for i in range(len(eig_val_sc))]
				
        if checker=='0':
            result= matrix.tolist()
            result=json.dumps(result) 	
        elif checker=='1':
            eig_pairs.sort()
            eig_pairs.reverse()
            matrix_w = np.hstack((eig_pairs[0][1].reshape(len(matrix),1),
                        eig_pairs[1][1].reshape(len(matrix),1)))
            transformed_matrix = matrix_w.T.dot(matrix)
            result= transformed_matrix.T
            result= result.tolist()
            result=json.dumps(result) 
            print(result)
        resp = make_response(result)
        resp.headers['Content-Type'] = "application/json"
        return resp
    return render_template('login.html', message='')
if __name__ == "__main__":
    app.run(debug = True)

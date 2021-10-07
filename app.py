from flask import Flask,render_template,request,redirect
import pickle as pkl
import numpy as np



app=Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/predict',methods=['GET','POST'])
def predict():
    team1=str(request.args.get('team1'))
    team2=str(request.args.get('team2'))
    if team1==team2:
        return redirect(url_for('index'))
    toss_winner=int(request.args.get('toss_winner'))
    choose=int(request.args.get('toss_decision'))
    print(team1,team2,toss_winner,choose)
    with open('/home/stellamarsh/ipl_project/ipl/model1.pkl','rb') as f:
        model1=pkl.load(f)

    with open('/home/stellamarsh/ipl_project/ipl/inv_vocab.pkl','rb') as f:
        inv_vocab=pkl.load(f)
    cteam1=inv_vocab[team1]
    cteam2=inv_vocab[team2]
    arr=np.array([cteam1,cteam2,choose,toss_winner]).reshape(1,-1)

    predict=model1.predict(arr)

    if predict==0:


        return render_template('after.html',data=team1)
    else:
        return render_template('after.html',data=team2)



if __name__=="__main__":
    app.run(debug=True)



from flask import Flask, request, render_template
from flask_cors import CORS
from script import upload_file


app = Flask(__name__)
CORS(app)
initial = 0 

@app.route('/kimathb', methods=['POST'])
def returne () :
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']
        form_data = request.form.to_dict()
        check = form_data['selected']   
        return  upload_file(file,check)
    return {"Nothing"}
    

@app.route('/')
def home():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)


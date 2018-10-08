from flask import Flask, request, jsonify
from pyspark.sql import SparkSession
import sys
import gc
import os
import json
import numpy as np
from keras.preprocessing import sequence
from keras.models import load_model
# Path for spark source folder
"""
os.environ['SPARK_HOME']="C:/Spark/"

# Append pyspark  to Python Path
sys.path.append("C:/Spark/python/")
"""
spark = SparkSession.builder.\
    master('local[*]') \
    .config("spark.executor.memory", "4g")\
                  .config("spark.driver.memory","18g")\
                  .config("spark.executor.cores","4")\
                  .config("spark.python.worker.memory","4g")\
                  .config("spark.driver.maxResultSize","0")\
                  .config("spark.default.parallelism","2")\
    .appName('ML_Deployment').getOrCreate()

sc = spark.sparkContext

app = Flask(__name__)

ErrorMSG ="""
Please sent json object as 
{ 
"Text": "This show doesn't even come up with its own topics to 'review'. The first episode of this dreck uses the same topics of its much funnier and, indeed, original counterpart. How is that original? Andy Daly is boring. His forced 'acting' style is so awkward I couldn't help but feel embarrassed for him. His 'lovely assistant' is just not necessary to the show and is clearly only there to be the token 'pretty girl' all shows in the US seem to require."
}
"""

@app.route('/ML', methods = ['POST'])
def postJsonHandler():
    #print (request.is_json)
    #Check JSON data
    if not request.is_json: 
        return ErrorMSG
    content = request.get_json()
    text = content['Text']
    if text is None:
        return ErrorMSG
    
    try:
        
        model = load_model('lstm_sent.h5')
        vocabs = np.load('vocabs.npy').item()
        #example="I can't properly express how pleased I am that this show turned out as well as it did. It's is extremely well shot, well written, and well acted."
        x = np.array([[vocabs[word] if word in vocabs.keys() else 0 for word in seq.split()] for seq in [text]])
        print ("Example Sentence: ", x)
        x_pad=sequence.pad_sequences(x,maxlen=100, padding='post')
        print ("example class(1 means positive, 0 means negative): ", (model.predict_classes(x_pad)))
        print ("example probability: ", model.predict_proba(x_pad))
        my_class = model.predict_classes(x_pad)
        my_prob =model.predict_proba(x_pad)
        classed =json.dumps(my_class.tolist())
        prob =json.dumps(my_prob.tolist())
        return jsonify(classed, prob)
    except Exception as ex:
        print(ex)

    return 'JSON posted'

@app.route('/')
def index():
    return "<h1>Project of cloud programming course</h1><br/><p>Mohammed Shehab</p> <p>Amir Farzad</p>"

@app.route('/user/<name>')
def user(name):
	return '<h1>Hello, {0}!</h1>'.format(name)

if __name__ == '__main__':
    app.run(port =5500, debug=True)
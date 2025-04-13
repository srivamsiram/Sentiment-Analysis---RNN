import numpy as np
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
model = load_model('model2.h5')
max_fatures = 2000
tokenizer = Tokenizer(num_words=max_fatures, split=' ')

def test(twt):    
    #vectorizing the tweet by the pre-fitted tokenizer instance
    tokenizer.fit_on_texts(twt)
    twt = tokenizer.texts_to_sequences(twt)
    #padding the tweet to have exactly the same shape as `embedding_2` input
    twt = pad_sequences(twt, maxlen=1504, dtype='int32', value=0)
    #print(twt)
    sentiment = model.predict(twt,batch_size=1,verbose = 2)[0]

    print(sentiment)
    if(np.argmax(sentiment) == 0):
        print("negative")
    elif (np.argmax(sentiment) == 1):
        print("positive")

twt = input()
while(twt!=""):
    test(twt)
    twt=input()

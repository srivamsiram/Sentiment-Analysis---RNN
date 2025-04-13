import os,time
import numpy as np
from tkinter import *
import time as t
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
print("Starting...")
window = Tk()
window.geometry("1000x550")
window.resizable(0,0)
window.title("Main Project")

input_name = StringVar()
input_review = StringVar()
input_result = StringVar()

def predict():
    print("Analysing..")    
    input_result.set("Analysing.............")
    result = ""
    name = input_name.get()
    result = "Model : "+name+"\n"
    input_name.set("")
    twt = input_review.get()
    input_review.set("")
    start = time.time()
    model = load_model('model2.h5')
    result = "Trained model Location : "+ os.getcwd() +"\n"
    max_fatures = 2000
    result = result+ "Max features = 2000 \n"
    result = result +"Text: \n"+ str(twt) + '\n'
    tokenizer = Tokenizer(num_words=max_fatures, split=' ')
    tokenizer.fit_on_texts(twt)
    twt = tokenizer.texts_to_sequences(twt)
    result = result +"Tokenized Text: \n"+ str(twt) + '\n'
    #padding the tweet to have exactly the same shape as `embedding_2` input
    twt = pad_sequences(twt, maxlen=1504, dtype='int32', value=0)
    #print(twt)
    sentiment = model.predict(twt,batch_size=1,verbose = 2)[0]
    print(sentiment)
    result += "Sentiment \n" +str(sentiment)+'\n'
    if (np.argmax(sentiment) == 0):
        print("negative")
        temp = "negative"
    elif (np.argmax(sentiment) == 1):
        print("positive")
        temp = "positive"
    end = time.time()
    diff = end-start
    result += "Time taken :" + str(abs(diff))+" sec\n"
    result += "Polarity :"+ temp +"\n"
    input_result.set(result)
    
label = Label(window,text="Sentiment Analysis on Product Reviews using RNN",fg="white",height =1,bg="black")
label.config(font=("Courier", 20))
label.pack(fill="x")

input_frame = Frame(window,width = 1000, height = 100, bd = 0,bg="#eee", highlightbackground = "black", highlightcolor = "black", highlightthickness = 1)
input_frame.pack(side = TOP)

btn_frame = Frame(window,width = 1000,height =50,bd = 0, bg="blue")
btn_frame.pack(side=BOTTOM)

result_frame = Frame(window,width = 600, height = 400, bd = 0,bg="yellow", highlightbackground = "black", highlightcolor = "black", highlightthickness = 1)
result_frame.pack(side =LEFT)

graph_frame = Frame(window,width = 400, height = 400, bd = 0, bg="red",highlightbackground = "black", highlightcolor = "black", highlightthickness = 1)
graph_frame.pack(side =RIGHT)

Name_label= Label(input_frame,text="Name of the Product ",font = ('arial', 18, 'bold'),width=30)
Name_label.grid(row=0)
input_field = Entry(input_frame, font = ('arial', 18, 'bold'), textvariable = input_name, width = 50, bg = "white", bd = 1, justify = LEFT)
input_field.grid(row = 0, column = 1)

review_label= Label(input_frame,text="Review",width=30,font = ('arial', 18, 'bold'))
review_label.grid(row=1)
review_input_field = Entry(input_frame, font = ('arial', 18, 'bold'), textvariable = input_review, width = 50, bg = "white", bd = 1, justify = LEFT)
review_input_field.grid(row = 1, column = 1)

Result_label = Label(result_frame,text="Result",bg="#eee",fg="black",width=60)
Result_label.config(font=("arial", 15))
Result_label.pack(side = "top",fill='x')

Res_label = Label(result_frame,text = " ",textvariable = input_result,bg="white",width=60,height=40,justify='left')
Res_label.config(font=("arial", 15))
Res_label.pack(side="bottom",fill="x")

graph_label = Label(graph_frame,text="Graph",bg="#eee",fg="black",width=40)
graph_label.config(font=("arial", 15))
graph_label.pack(side = "top",fill='x')

graph_Res_label = Label(graph_frame,text = " ",bg="white",width=40,height = 40)
graph_Res_label.pack(side="bottom",fill="x")

btn = Button(btn_frame,text = "Analyse", command = lambda: predict()).pack(side="right")

window.mainloop()

import tkinter as tk
from tkinter.ttk import *
import cv2
from pyzbar.pyzbar import decode
from PIL import Image, ImageTk
import pytesseract
import numpy as np
import pandas as pd
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

from tkinter import messagebox
from tkinter import filedialog
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import EfficientNetB0


pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'

def Automatic_page(master,selected_folder,text_list, barcode_list, filepath1, filepath2, filepath3):
    text = ""
    search_code=""

    def preprocess(text):
        # Tokenize text
        tokens = word_tokenize(text.lower())
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
        # Lemmatize tokens
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        return " ".join(tokens)

    def calculate_similarity(paragraph1, paragraph2):
        # Preprocess both paragraphs
        p1 = preprocess(paragraph1)
        p2 = preprocess(paragraph2)
        
        # Vectorize the paragraphs using CountVectorizer
        vectorizer = CountVectorizer().fit([p1, p2])
        vectors = vectorizer.transform([p1, p2]).toarray()
        
        # Calculate Cosine Similarity
        similarity = cosine_similarity(vectors)[0][1]
        # Convert similarity to percentage
        similarity_percentage = round(similarity * 100, 2)
        return similarity_percentage

    def Result_similarity(paragraph):
        # Compare each pair of text files
        sscore=[]
        if len(text_list) > 1:
            for i in range(len(text_list)):
                similarity_score = calculate_similarity(text_list[i], paragraph)
                print(f"Similarity between File {i + 1} and paragraph: {similarity_score:.2f}%")
                sscore.append(round(similarity_score,2))
                
        else:
            print("Not enough files to compare.")

        return max(sscore)

    def Barcode_similarity(search_code):
        if barcode_list or search_code:
            # Search for the barcode
            if search_code in barcode_list:
                return "Match",100
            else:
                return "Not Match",0
        else:
            return "Invalid",0

    def DetectLogo(selected_foldertitle,filepathselect):
        # Load the saved model
        resultstr=""
        confidencestr="0.0"
        confidenceflot=0.0
        
        model = keras.models.load_model('Detectionmodel_11.h5')

        # Open a file dialog to select an image for detection
        file_path = filepathselect
        if file_path:
            # Load and preprocess the image
            img = image.load_img(file_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = keras.applications.efficientnet.preprocess_input(img_array)
            img_array = np.expand_dims(img_array, axis=0)

            # Predict the class of the image
            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions, axis=1)
            confidence = np.max(predictions)
            
            # Decode the predicted class to the original label
            label_encoder = LabelEncoder()
            label_encoder.fit(os.listdir('Product'))
            predicted_label = label_encoder.inverse_transform(predicted_class)

            #messagebox.showinfo("Detection Result", f"The detected logo is: {predicted_label[0]}")
            #messagebox.showinfo("Detection Result", f"The detected logo is: {predicted_label[0]}\nConfidence: {confidence:.2f}")
            print(selected_foldertitle,'---',predicted_label[0])
            
            if selected_foldertitle==predicted_label[0]:
                resultstr="\nDetected Logo : "+str(predicted_label[0])
                confidencestr="Match Confidence: "+str(round(confidence*100,2))
                confidenceflot=round(confidence,2)*100
                if confidence<=0.70:
                    #resultstr="\nDetected Logo Fail: "+str(predicted_label[0])
                    resultstr="\nDetected Logo Fail "
                    #confidencestr="Match Confidence: "+str(round(confidence*100,2))
                    #confidenceflot=round(confidence,2)*100
            else:
                #resultstr="\nDetected Logo Fail: "+str(predicted_label[0])
                resultstr="\nDetected Logo Fail "
                #confidencestr="Match Confidence: "+str(round(confidence*100,2))
                #confidenceflot=round(confidence*100,2)
                
        return resultstr,confidencestr,confidenceflot


    # Create a new window
    root = tk.Toplevel(master)
    root.title("Fake Product Detection")
    root.geometry("1000x600")
    root.resizable(True, True)

    # Configure layout
    root.columnconfigure(0, weight=1)
    root.columnconfigure(1, weight=1)
    root.columnconfigure(2, weight=1)
    root.rowconfigure(1, weight=1)

    # Header Label
    header_label = tk.Label(
        root,
        text="Fake Product Detection",
        font=("Arial", 24, "bold"),
        bg="black",
        fg="white",
        anchor="center",
    )
    #header_label.grid(row=0, column=0, columnspan=3, sticky="ew", padx=10, pady=10)

    countval=0;
    Barcodeval=0;
    OCRval=0;
    LOGOval=0;   


    #Barcode'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    if filepath1!='':
        countval=countval+1
        # Barcode Processing
        frame1 = cv2.imread(filepath1)
        gray_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        barcodes = decode(gray_frame1)
        for barcode in barcodes:
            data = barcode.data.decode('utf-8')
            print("+++++++++++++++++++++++++++++++++++++++++")
            print(data)
            print("+++++++++++++++++++++++++++++++++++++++++")
            search_code = data
            points = barcode.polygon
            if len(points) >= 4:
                pts = [(point.x, point.y) for point in points]
                cv2.polylines(frame1, [np.array(pts)], True, (0, 255, 0), 3)
            cv2.putText(
                frame1,
                f"Barcode: {data}",
                (barcode.rect.left, barcode.rect.top - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )
        resized_frame1 = cv2.resize(frame1, (300, 300))
        img1 = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(resized_frame1, cv2.COLOR_BGR2RGB)))

        # Barcode Image Label
        camera_label1 = tk.Label(root, image=img1, borderwidth=2, relief="solid")
        camera_label1.grid(row=1, column=0, padx=10, pady=10)
        camera_label1.image = img1

        # Barcode Result Labels
        barcode_label = tk.Label(root, text=f"Barcode: {search_code}", font=("Arial", 14), bg="#0000FF", fg="white")
        barcode_label.grid(row=2, column=0, padx=10, pady=10)

        
        Rval1 = tk.StringVar()
        Rlabel1 = tk.Label(root ,width=20,font=("Arial", 14), bg="#0000FF", fg="white",textvariable=Rval1).grid(row=3, column=0)
        Barcodresult,Barcodeval=Barcode_similarity(search_code)
        Rval1.set("Barcode Result-"+Barcodresult)

    #OCR ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    if filepath2!='':
        countval=countval+1
        # OCR Processing
        frame2 = cv2.imread(filepath2)
        gray_frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray_frame2, lang="eng").strip()
        print("+++++++++++++++++++++++++++++++++++++++++")
        print(text)
        print("+++++++++++++++++++++++++++++++++++++++++")
        resized_frame2 = cv2.resize(frame2, (300, 300))
        img2 = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(resized_frame2, cv2.COLOR_BGR2RGB)))

        # OCR Image Label
        camera_label2 = tk.Label(root, image=img2, borderwidth=2, relief="solid")
        camera_label2.grid(row=1, column=1, padx=10, pady=10)
        camera_label2.image = img2

        # OCR Result Labels
        ocr_label = tk.Label(root, text=f"OCR Text: {text[:30]}...", font=("Arial", 14), bg="#0000FF", fg="white")
        ocr_label.grid(row=2, column=1, padx=10, pady=10)

        Rval2 = tk.StringVar()
        Rlabel2 = tk.Label(root ,width=20,font=("Arial", 14), bg="#0000FF", fg="white",textvariable=Rval2).grid(row=3, column=1)
        OCRval=Result_similarity(text)
        #decimal_digits = str(OCRval).split(".")[1][:2].ljust(2, '0')
        Rval2.set("Text Result - " + str(OCRval)+"%")
        



    #IMG logo''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    if filepath3!='':
        countval=countval+1
        # IMG Processing
        frame3 = cv2.imread(filepath3)
        resized_frame3 = cv2.resize(frame3, (300, 300))
        img3 = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(resized_frame3, cv2.COLOR_BGR2RGB)))

        # IMG Image Label
        camera_label3 = tk.Label(root, image=img3, borderwidth=2, relief="solid")
        camera_label3.grid(row=1, column=2, padx=10, pady=10)
        camera_label3.image = img3
        
        resultstr1,confidencestr1,confidenceflot=DetectLogo(selected_folder,filepath3)
        #print(resultstr1)
        # IMG Result Labels
        IMG_label = tk.Label(root, text=f"Product {resultstr1}", font=("Arial", 14), bg="#0000FF", fg="white")
        IMG_label.grid(row=2, column=2, padx=10, pady=10)

        Rval3 = tk.StringVar()
        Rlabel3 = tk.Label(root ,width=20,font=("Arial", 14), bg="#0000FF", fg="white",textvariable=Rval3).grid(row=3, column=2)

        LOGOval=confidenceflot;
        Rval3.set(""+str(confidencestr1)+"%")


    #LOGOval=LOGOval*100
    #OCRval=OCRval*100
    print(Barcodeval,", ",OCRval,", ",LOGOval,", ",countval)
    
    finalResult=int((Barcodeval+OCRval+LOGOval)/countval)

    Rval4 = tk.StringVar()
    placeholder_label = tk.Label(root,width=40, textvariable=Rval4, font=("Arial", 24), bg="#0000FF", fg="white")
    placeholder_label.grid(row=4, column=0, columnspan=3, padx=10, pady=10)
    Rval4.set("Result-"+str(finalResult)+"%")
    
    Rval5 = tk.StringVar()
    # Placeholder for Additional Features
    placeholder_label1 = tk.Label(root,width=40, textvariable=Rval5, font=("Arial", 24), bg="#0000FF", fg="white")
    placeholder_label1.grid(row=5, column=0, columnspan=3, padx=10, pady=10)
    if finalResult>=75:
        Rval5.set("Quality Product")
    else:
        Rval5.set("Fake Product.")
    
    # Start the GUI loop
    root.mainloop()

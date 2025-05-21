from tkinter.ttk import *
import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog
import pandas as pd
import numpy as np
import Detect
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import cv2
import time
import threading
from datetime import datetime, timedelta
'''
# Download necessary NLTK data
nltk.download('punkt')
#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('wordnet')

'''

cap = cv2.VideoCapture(0)
width, height = 400, 300
selected_folder=""
framecapture=0;

def read_barcodes_from_file(file_path):
    """
    Reads barcodes from a text file and stores them in a list.
    """
    try:
        with open(file_path, 'r') as file:
            barcodes = [line.strip() for line in file if line.strip()]
        return barcodes
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return []

def read_text_files_from_folder(folder_path):
    """
    Reads all text files from a folder and stores their content in a list.
    """
    text_contents = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                text_contents.append(file.read())
    return text_contents

# Function to fetch folder list from the 'Product' folder
def update_folder_list():
    product_folder = "Product"  # Specify the path to the Product folder
    if not os.path.exists(product_folder):
        os.makedirs(product_folder)  # Create the folder if it doesn't exist
    folders = [f for f in os.listdir(product_folder) if os.path.isdir(os.path.join(product_folder, f))]
    folder_combo['values'] = folders  # Update combo box values
    if folders:
        folder_combo.current(0)  # Select the first folder by default

# Function to handle folder selection
def on_folder_select(event):
    global selected_folder
    selected_folder = folder_combo.get()
    print(f"Selected Folder: {selected_folder}")  # Do something with the selected folder


def file_opener():
    input11 =filedialog.askopenfilename(initialdir = "/",title = "Select a File",filetypes = (("[png]","*.png*"),("[PNG]","*.PNG*"),("[jpg]","*.jpg*"),("[JPG]","*.JPG*"),("all files","*.*")))
    filepath.set(input11)

def file_opener1(ss):
    input11 =filedialog.askopenfilename(initialdir = "/",title = "Select a File",filetypes = (("[png]","*.png*"),("[PNG]","*.PNG*"),("[jpg]","*.jpg*"),("[JPG]","*.JPG*"),("all files","*.*")))
    if ss==1:
        filepath1.set(input11)
    if ss==2:
        filepath2.set(input11)
    if ss==3:
        filepath3.set(input11)

def cam_capture(a):
    global framecapture
    framecapture=a
        
def Automatic_detect():
    print(selected_folder)
    # Example usage
    file_path = "Product/"+selected_folder+"/barcodes.txt"  # Replace with your file path
    # Read barcodes from file
    barcode_list = read_barcodes_from_file(file_path)
    print(barcode_list)
    # Read all text files
    folder_path = "Product/"+selected_folder+"/Text"
    text_list = read_text_files_from_folder(folder_path)
    print(text_list)

    Detect.Automatic_page(master,selected_folder,text_list,barcode_list,str(filepath1.get()),str(filepath2.get()),str(filepath3.get()))


master = tk.Tk()
master.title("Fake Product Detection System")
master.geometry("1200x500")
master.resizable(False, False)
#master.configure(background='#000000')

label = tk.Label(master ,width=52,text = "Fake Product Detection System",font=("arial italic", 30), bg="#000000", fg="white").grid(row=0, column=0,columnspan=4)

# Combo box for folder list
folder_label = tk.Label(master, text="Select Product:", font=("arial", 15)).grid(row=1, column=0, pady=10)
folder_combo = Combobox(master, font=("arial", 12), width=25)
folder_combo.grid(row=1, column=1, pady=10)
folder_combo.bind("<<ComboboxSelected>>", on_folder_select)  # Bind selection event
update_folder_list()

btnd1 = tk.Button(master,text="Select Barcode Image",font=("arial italic", 15), bg="#0000FF", fg="white",width=25,command=lambda:file_opener1(1)).grid(row=2, column=0,padx=1, pady=20)
filepath1 = tk.StringVar()
aa11 = tk.Entry(master,font=("arial italic", 15), bg="white", fg="Blue",width=25,textvariable=filepath1).grid(row=2, column=1,padx=1, pady=1)
#filepath1.set("E:/Project 2024/fake Product detect/barcode2.jpeg")

btnd2 = tk.Button(master,text="Select Text Image",font=("arial italic", 15), bg="#0000FF", fg="white",width=25,command=lambda:file_opener1(2)).grid(row=3, column=0,padx=1, pady=20)
filepath2 = tk.StringVar()
aa12 = tk.Entry(master,font=("arial italic", 15), bg="white", fg="Blue",width=25,textvariable=filepath2).grid(row=3, column=1,padx=1, pady=1)
#filepath2.set("E:/Project 2024/fake Product detect/FLOW.PNG")

btnd3 = tk.Button(master,text="Select Logo Image",font=("arial italic", 15), bg="#0000FF", fg="white",width=25,command=lambda:file_opener1(3)).grid(row=4, column=0,padx=1, pady=20)
filepath3 = tk.StringVar()
aa13 = tk.Entry(master,font=("arial italic", 15), bg="white", fg="Blue",width=25,textvariable=filepath3).grid(row=4, column=1,padx=1, pady=1)
#filepath3.set("E:/Project 2024/fake Product detect/File/addidaslogo.png")


btn1 = tk.Button(master,text="Start Detect",font=("arial italic", 15), bg="#000000", fg="white",width=20,command=lambda:Automatic_detect()).grid(row=5, column=0,padx=1, pady=20)

btn2 = tk.Button(master,text="Exit",font=("arial italic", 15), bg="#000000", fg="white",width=20,command=master.destroy).grid(row=5, column=1,padx=1, pady=20)

btncam1 = tk.Button(master,text="Cam",font=("arial italic", 15), bg="#FF0033", fg="white",width=8,command=lambda:cam_capture(1)).grid(row=2, column=2,padx=1, pady=2)
btncam2 = tk.Button(master,text="Cam",font=("arial italic", 15), bg="#FF0033", fg="white",width=8,command=lambda:cam_capture(2)).grid(row=3, column=2,padx=1, pady=2)
btncam3 = tk.Button(master,text="Cam",font=("arial italic", 15), bg="#FF0033", fg="white",width=8,command=lambda:cam_capture(3)).grid(row=4, column=2,padx=1, pady=2)


# Display live camera feed
camera_label = tk.Label(master,width=400, height=300, borderwidth=2, relief="solid")
camera_label.grid(row=1, column=3, rowspan=5, padx=10, pady=10)

#Start live camera feed
def show_camera():
    global framecapture
    while True:
        ret, frame = cap.read()
        if ret:
            frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2image = cv2.resize(frame, (width, height))
            #cv2image = cv2.flip(frame, 1)
            if framecapture==1:
                unique_filename = f"Image/ImgBarcode_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                cv2.imwrite(unique_filename, cv2image)
                filepath1.set(unique_filename)
                framecapture=0
            if framecapture==2:
                unique_filename = f"Image/ImgText_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                cv2.imwrite(unique_filename, cv2image)
                filepath2.set(unique_filename)
                framecapture=0
            if framecapture==3:
                unique_filename = f"Image/ImgLogo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                cv2.imwrite(unique_filename, cv2image)
                filepath3.set(unique_filename)
                framecapture=0
                
 
            cv2image = cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGB)
            img = tk.PhotoImage(data=cv2.imencode('.png', cv2image)[1].tobytes())
            camera_label.config(image=img)
            camera_label.image = img
        else:
            break

    if cv2.waitKey(1) & 0xFF == ord('a'):
        master.quit()
        
    camera_label.after(10, show_camera)
        

t = threading.Thread(target=show_camera)
t.start()

master.mainloop()

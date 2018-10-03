"""
****************************************
Create a folder named "Original_image"
And put the carrier image in that folder
To get more information see line no. 75
****************************************
"""
import os
import shutil
from PIL import Image
from pathlib import Path

#encoding part :
def encode_image(img, msg):
    length = len(msg)
    if length > 255:
        print("text too long! (don't exeed 255 characters)")
        return False
    encoded = img.copy()
    width, height = img.size
    index = 0
    for row in range(height):
        for col in range(width):
            if img.mode != 'RGB':
                r, g, b ,a = img.getpixel((col, row))
            elif img.mode == 'RGB':
                r, g, b = img.getpixel((col, row))
            # first value is length of msg
            if row == 0 and col == 0 and index < length:
                asc = length
            elif index <= length:
                c = msg[index -1]
                asc = ord(c)
            else:
                asc = r
            encoded.putpixel((col, row), (asc, g , b))
            index += 1
    return encoded

#decoding part :
def decode_image(img):
    width, height = img.size
    msg = ""
    index = 0
    for row in range(height):
        for col in range(width):
            if img.mode != 'RGB':
                r, g, b ,a = img.getpixel((col, row))
            elif img.mode == 'RGB':
                r, g, b = img.getpixel((col, row))	
            # first pixel r value is length of message
            if row == 0 and col == 0:
                length = r
            elif index <= length:
                msg += chr(r)
            index += 1
    decoded_image_file = "decoded_image.png"
    img.save(decoded_image_file)
    print("Decoded image was saved!")
    return msg

#driver part :
#deleting previous folders :
if os.path.exists("Encoded_image/"):
    shutil.rmtree("Encoded_image/")
if os.path.exists("Decoded_output/"):
    shutil.rmtree("Decoded_output/")
#creating new folders :
os.makedirs("Encoded_image/")
os.makedirs("Decoded_output/")
while True:
    m = input("To encode press '1', to decode press '2', press any other button to close: ")
    if m == "1":
        name_of_file = input("Enter the name of the file with extension : ")
        original_image_file = "Original_image/"+name_of_file 
        img = Image.open(original_image_file)
        print("Description : ",img,"\nMode : ", img.mode)
        encoded_image_file = "encoded_image.png"
        secret_msg = input("Enter the message you want to hide: ")
        print("The message length is: ",len(secret_msg))
        img_encoded = encode_image(img, secret_msg)
        os.chdir("Encoded_image/")
        img_encoded.save(encoded_image_file) # saving the image with the hidden text
        print("Encoded image was saved!")
    elif m == "2":
        encoded_image_file = "encoded_image.png"
        img = Image.open(encoded_image_file)
        os.chdir("..") #going back to parent directory
        os.chdir("Decoded_output/")
        hidden_text = decode_image(img) 
        file = open("hidden_text.txt","w")
        file.write(hidden_text) # saving hidden text as text file :
        file.close()
        print("Hidden text saved as text file")
    else:
        print("Closed!")
        break
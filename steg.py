import os
import shutil
import cv2
import sys
import numpy as np
import itertools
from PIL import Image
from pathlib import Path

quant = np.array([[16,11,10,16,24,40,51,61], # required for DCT
                    [12,12,14,19,26,58,60,55],
                    [14,13,16,24,40,57,69,56],
                    [14,17,22,29,51,87,80,62],
                    [18,22,37,56,68,109,103,77],
                    [24,35,55,64,81,104,113,92],
                    [49,64,78,87,103,121,120,101],
                    [72,92,95,98,112,100,103,99]])

#encoding part :
def toBits(msg):
    bits = []

    for char in msg:
        binval = bin(ord(char))[2:].rjust(8,'0')
        
        #for bit in binval: 
        bits.append(binval)

    self.numBits = bin(len(bits))[2:].rjust(8,'0')
    return bits
def chunks(l, n): # required for DCT
        m = int(n)
        for i in range(0, len(l), m):
            yield l[i:i + m]
def dct_encode_image(img,msg):
    bitMess = toBits(msg)
    row,col = img.shape[:2]
    if((col/8)*(row/8)<len(secret)):
        print("DCT: Message too large to encode in image")
        return 
    if row%8 != 0 or col%8 != 0:
        img = cv2.resize(img,(col+(8-col%8),row+(8-row%8)))
    row,col = img.shape[:2]
    #split image into RGB channels
    bImg,gImg,rImg = cv2.split(img)
    #message to be hid in blue channel so converted to type float32 for dct function
    bImg = np.float32(bImg)
    #break into 8x8 blocks
    imgBlocks = [np.round(bImg[j:j+8, i:i+8]-128) for (j,i) in itertools.product(range(0,row,8),
                                                                       range(0,col,8))]
    #Blocks are run through DCT function
    dctBlocks = [np.round(cv2.dct(img_Block)) for img_Block in imgBlocks]
    #blocks then run through quantization table
    quantizedDCT = [np.round(dct_Block/quant) for dct_Block in dctBlocks]
    #set LSB in DC value corresponding bit of message
    messIndex = 0
    letterIndex = 0
    for quantizedBlock in quantizedDCT:
        #find LSB in DC coeff and replace with message bit
        DC = quantizedBlock[0][0]
        DC = np.uint8(DC)
        DC = np.unpackbits(DC)
        
        DC[7] = bitMess[messIndex][letterIndex]
        DC = np.packbits(DC)
        
        DC = np.float32(DC)
        DC= DC-255
        quantizedBlock[0][0] = DC

        letterIndex = letterIndex+1
        if letterIndex == 8:
            letterIndex = 0
            messIndex = messIndex + 1
            if messIndex == len(msg):
                break

    #blocks run inversely through quantization table
    sImgBlocks = [quantizedBlock *quant+128 for quantizedBlock in quantizedDCT]
    #puts the new image back together
    sImg=[]
    for chunkRowBlocks in chunks(sImgBlocks, col/8):
        for rowBlockNum in range(8):
            for block in chunkRowBlocks:
                sImg.extend(block[rowBlockNum])
    sImg = np.array(sImg).reshape(row, col)
    #converted from type float32
    sImg = np.uint8(sImg)
    sImg = cv2.merge((sImg,gImg,rImg))
    return sImg

def dwt_encode_image(img,msg):


    return encoded

def lsb_encode_image(img, msg):
    length = len(msg)
    if length > 255:
        print("LSB: text too long! (don't exeed 255 characters)")
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

def dct_encode_image(img):
    row,col = img.shape[:2]

    messSize = None
    messageBits = []
    buff = 0
    #split image into RGB channels
    bImg,gImg,rImg = cv2.split(img)
    #message hid in blue channel so converted to type float32 for dct function
    bImg = np.float32(bImg)
    #break into 8x8 blocks
    imgBlocks = [bImg[j:j+8, i:i+8]-128 for (j,i) in itertools.product(range(0,row,8),
                                                                       range(0,col,8))]  
    #blocks run through quantization table
    quantizedDCT = [img_Block/quant for img_Block in imgBlocks]
    i=0
    #message extracted from LSB of DC coeff
    for quantizedBlock in quantizedDCT:
        DC = quantizedBlock[0][0]
        DC = np.uint8(DC)
        DC = np.unpackbits(DC)
        if DC[7] == 1:
            buff+=(0 & 1) << (7-i)
        elif DC[7] == 0:
            buff+=(1&1) << (7-i)
        i=1+i
        if i == 8:
            messageBits.append(chr(buff))
            buff = 0
            i =0
            if messageBits[-1] == '*' and messSize is None:
                try:
                    messSize = int(''.join(messageBits[:-1]))
                except:
                    pass
        if len(messageBits) - len(str(messSize)) - 1 == messSize:
            return ''.join(messageBits)[len(str(messSize))+1:]

    img.save(img)
    return ''

def dwt_encode_image(img):



    img.save(img)
    return msg

def lsb_decode_image(img):
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
    img.save(img)
    print("Decoded images were saved!")
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
original_image_file # to make the file name global variable
while True:
    m = input("To encode press '1', to decode press '2', press any other button to close: ")
    if m == "1":
        name_of_file = input("Enter the name of the file with extension : ")
        original_image_file = "Original_image/"+name_of_file 
        img = Image.open(original_image_file)
        print("Description : ",img,"\nMode : ", img.mode)
        secret_msg = input("Enter the message you want to hide: ")
        print("The message length is: ",len(secret_msg))
        lsb_img_encoded = lsb_encode_image(img, secret_msg)
        dct_img_encoded = dct_encode_image(img, secret_msg)
        dwt_img_encoded = dwt_encode_image(img, secret_msg)
        os.chdir("Encoded_image/")
        lsb_encoded_image_file = "lsb_" + original_image_file
        lsb_img_encoded.save(lsb_encoded_image_file)
        dct_encoded_image_file = "dct_" + original_image_file
        dct_img_encoded.save(dct_encoded_image_file)
        dwt_encoded_image_file = "dwt_" + original_image_file
        dwt_img_encoded.save(dwt_encoded_image_file) # saving the image with the hidden text
        print("Encoded images were saved!")
    elif m == "2":
        lsb_encoded_image_file = "lsb_" + original_image_file
        lsb_img = Image.open(lsb_encoded_image_file)
        dct_encoded_image_file = "dct_" + original_image_file
        dct_img = Image.open(dct_encoded_image_file)
        dwt_encoded_image_file = "dwt_" + original_image_file
        dwt_img = Image.open(dwt_encoded_image_file)
        os.chdir("..") #going back to parent directory
        os.chdir("Decoded_output/")
        lsb_hidden_text = lsb_decode_image(lsb_img)
        dct_hidden_text = dct_decode_image(dct_img) 
        dwt_hidden_text = dwt_decode_image(dwt_img) 
        file = open("lsb_hidden_text.txt","w")
        file.write(lsb_hidden_text) # saving hidden text as text file
        file.close()
        file = open("dct_hidden_text.txt","w")
        file.write(dct_hidden_text) # saving hidden text as text file
        file.close()
        file = open("dwt_hidden_text.txt","w")
        file.write(dwt_hidden_text) # saving hidden text as text fikle
        file.close()
        print("Hidden text saved as text file !")
    else:
        print("Closed!")
        break
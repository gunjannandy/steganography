import os
import shutil
import cv2
import sys
import numpy as np
import itertools
import PIL.Image
from pathlib import Path
quant = np.array([[16,11,10,16,24,40,51,61],   # (QUANTIZATION TABLE)
                    [12,12,14,19,26,58,60,55], # required for DCT
                    [14,13,16,24,40,57,69,56],
                    [14,17,22,29,51,87,80,62],
                    [18,22,37,56,68,109,103,77],
                    [24,35,55,64,81,104,113,92],
                    [49,64,78,87,103,121,120,101],
                    [72,92,95,98,112,100,103,99]])
class DCT():    
    def __init__(self): # Constructor
        self.message = None
        self.bitMess = None
        self.oriCol = 0
        self.oriRow = 0
        self.numBits = 0   
    #encoding part : 
    def encode_image(self,img,secret_msg):
        secret=secret_msg
        self.message = str(len(secret))+'*'+secret
        self.bitMess = self.toBits()
        #get size of image in pixels
        row,col = img.shape[:2]
        self.oriRow, self.oriCol = row, col  
        if((col/8)*(row/8)<len(secret)):
            print("Error: Message too large to encode in image")
            return False
        #make divisible by 8x8
        if row%8 != 0 or col%8 != 0:
            img = self.addPadd(img, row, col)
        
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
            DC[7] = self.bitMess[messIndex][letterIndex]
            DC = np.packbits(DC)
            DC = np.float32(DC)
            DC= DC-255
            quantizedBlock[0][0] = DC
            letterIndex = letterIndex+1
            if letterIndex == 8:
                letterIndex = 0
                messIndex = messIndex + 1
                if messIndex == len(self.message):
                    break
        #blocks run inversely through quantization table
        sImgBlocks = [quantizedBlock *quant+128 for quantizedBlock in quantizedDCT]
        #blocks run through inverse DCT
        #sImgBlocks = [cv2.idct(B)+128 for B in quantizedDCT]
        #puts the new image back together
        sImg=[]
        for chunkRowBlocks in self.chunks(sImgBlocks, col/8):
            for rowBlockNum in range(8):
                for block in chunkRowBlocks:
                    sImg.extend(block[rowBlockNum])
        sImg = np.array(sImg).reshape(row, col)
        #converted from type float32
        sImg = np.uint8(sImg)
        sImg = cv2.merge((sImg,gImg,rImg))
        return sImg

    #decoding part :
    def decode_image(img):
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
        #quantizedDCT = [dct_Block/ (quant) for dct_Block in dctBlocks]
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
        #blocks run inversely through quantization table
        sImgBlocks = [quantizedBlock *quant+128 for quantizedBlock in quantizedDCT]
        #blocks run through inverse DCT
        #sImgBlocks = [cv2.idct(B)+128 for B in quantizedDCT]
        #puts the new image back together
        sImg=[]
        for chunkRowBlocks in self.chunks(sImgBlocks, col/8):
            for rowBlockNum in range(8):
                for block in chunkRowBlocks:
                    sImg.extend(block[rowBlockNum])
        sImg = np.array(sImg).reshape(row, col)
        #converted from type float32
        sImg = np.uint8(sImg)
        sImg = cv2.merge((sImg,gImg,rImg))
        sImg.save(img)
        return ''
      
    """Helper function to 'stitch' new image back together"""
    def chunks(self,l, n):
        m = int(n)
        for i in range(0, len(l), m):
            yield l[i:i + m]
    def addPadd(self,img, row, col):
        img = cv2.resize(img,(col+(8-col%8),row+(8-row%8)))    
        return img
    def toBits(self):
        bits = []
        for char in self.message:
            binval = bin(ord(char))[2:].rjust(8,'0')
            bits.append(binval)
        self.numBits = bin(len(bits))[2:].rjust(8,'0')
        return bits

class LSB():
    #encoding part :
    def encode_image(self,img, msg):
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
    def decode_image(self,img):
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
        print("{} was saved!".format(decoded_image_file))
        return msg

#class DWT():
    #encoding part :

    #decoding part :


#driver part :
#deleting previous folders :
if os.path.exists("Encoded_image/"):
    shutil.rmtree("Encoded_image/")
if os.path.exists("Decoded_output/"):
    shutil.rmtree("Decoded_output/")
#creating new folders :
os.makedirs("Encoded_image/")
os.makedirs("Decoded_output/")
original_image_file = "" # to make the file name global variable
#creating instances from classses
#lsb = LSB()
#dct = DCT()
#dwt = DWT()
while True:
    m = input("To encode press '1', to decode press '2', to compare press '3', press any other button to close: ")

    if m == "1":
        name_of_file = input("Enter the name of the file with extension : ")
        original_image_file = "Original_image/"+name_of_file 
        img = PIL.Image.open(original_image_file)
        print("Description : ",img,"\nMode : ", img.mode)
        secret_msg = input("Enter the message you want to hide: ")
        print("The message length is: ",len(secret_msg))
        os.chdir("Encoded_image/")
        lsb_img_encoded = LSB().encode_image(img, secret_msg)
        dct_img_encoded = DCT().encode_image(img, secret_msg)
        #dwt_img_encoded = dwt.encode_image(img, secret_msg)
        lsb_encoded_image_file = "lsb_" + original_image_file
        lsb_img_encoded.save(lsb_encoded_image_file)
        dct_encoded_image_file = "dct_" + original_image_file
        dct_img_encoded.save(dct_encoded_image_file)
        #dwt_encoded_image_file = "dwt_" + original_image_file
        #dwt_img_encoded.save(dwt_encoded_image_file) # saving the image with the hidden text
        print("Encoded images were saved!")

    elif m == "2":
        lsb_encoded_image_file = "lsb_" + original_image_file
        lsb_img = PIL.Image.open(lsb_encoded_image_file)
        dct_encoded_image_file = "dct_" + original_image_file
        dct_img = PIL.Image.open(dct_encoded_image_file)
        #dwt_encoded_image_file = "dwt_" + original_image_file
        #dwt_img = PIL.Image.open(dwt_encoded_image_file)
        os.chdir("..") #going back to parent directory
        os.chdir("Decoded_output/")
        lsb_hidden_text = LSB().decode_image(lsb_img)
        dct_hidden_text = DCT().decode_image(dct_img) 
        #dwt_hidden_text = DWT().decode_image(dwt_img) 
        file = open("lsb_hidden_text.txt","w")
        file.write(lsb_hidden_text) # saving hidden text as text file
        file.close()
        file = open("dct_hidden_text.txt","w")
        file.write(dct_hidden_text) # saving hidden text as text file
        file.close()
        #file = open("dwt_hidden_text.txt","w")
        #file.write(dwt_hidden_text) # saving hidden text as text file
        #file.close()
        print("Hidden text saved as text file !")
    #elif m == "3":

        #comparison calls

    else:
        print("Closed!")
        break
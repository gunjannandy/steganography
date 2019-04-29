import os
import xlwt
import shutil
import cv2
import sys
import math
import numpy as np
import itertools
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
#from scipy import signal
quant = np.array([[16,11,10,16,24,40,51,61],      # QUANTIZATION TABLE
                    [12,12,14,19,26,58,60,55],    # required for DCT
                    [14,13,16,24,40,57,69,56],
                    [14,17,22,29,51,87,80,62],
                    [18,22,37,56,68,109,103,77],
                    [24,35,55,64,81,104,113,92],
                    [49,64,78,87,103,121,120,101],
                    [72,92,95,98,112,100,103,99]])
'''def show(im):
    im_resized = cv2.resize(im, (500, 500), interpolation=cv2.INTER_LINEAR)
    plt.imshow(cv2.cvtColor(im_resized, cv2.COLOR_BGR2RGB))
    plt.show()'''

'''
class DWT():   
    #encoding part : 
    def encode_image(self,img,secret_msg):
        #show(img)
        #get size of image in pixels
        row,col = img.shape[:2]
        #addPad
        if row%8 != 0 or col%8 != 0:
            img = cv2.resize(img,(col+(8-col%8),row+(8-row%8)))
        bImg,gImg,rImg = cv2.split(img)
        bImg = self.iwt2(bImg)
        #get size of paddded image in pixels
        height,width = bImg.shape[:2]
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


        return sImg

    #decoding part :
    def decode_image(self,img):
        msg = ""
        #get size of image in pixels
        row,col = img.shape[:2]
        bImg,gImg,rImg = cv2.split(img)

        return msg
      
    """Helper function to 'stitch' new image back together"""
    def _iwt(self,array):
        output = np.zeros_like(array)
        nx, ny = array.shape
        x = nx // 2
        for j in xrange(ny):
            output[0:x,j] = (array[0::2,j] + array[1::2,j])//2
            output[x:nx,j] = array[0::2,j] - array[1::2,j]
        return output

    def _iiwt(self,array):
        output = np.zeros_like(array)
        nx, ny = array.shape
        x = nx // 2
        for j in xrange(ny):
            output[0::2,j] = array[0:x,j] + (array[x:nx,j] + 1)//2
            output[1::2,j] = output[0::2,j] - array[x:nx,j]
        return output

    def iwt2(self,array):
        return _iwt(_iwt(array.astype(int)).T).T

    def iiwt2(self,array):
        return _iiwt(_iiwt(array.astype(int).T).T)


        '''

class DCT():    
    def __init__(self): # Constructor
        self.message = None
        self.bitMess = None
        self.oriCol = 0
        self.oriRow = 0
        self.numBits = 0   
    #encoding part : 
    def encode_image(self,img,secret_msg):
        #show(img)
        secret=secret_msg
        self.message = str(len(secret))+'*'+secret
        self.bitMess = self.toBits()
        #get size of image in pixels
        row,col = img.shape[:2]
        ##col, row = img.size
        self.oriRow, self.oriCol = row, col  
        if((col/8)*(row/8)<len(secret)):
            print("Error: Message too large to encode in image")
            return False
        #make divisible by 8x8
        if row%8 != 0 or col%8 != 0:
            img = self.addPadd(img, row, col)
        
        row,col = img.shape[:2]
        ##col, row = img.size
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
        #show(sImg)
        sImg = cv2.merge((sImg,gImg,rImg))
        return sImg

    #decoding part :
    def decode_image(self,img):
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
        ##sImg.save(img)
        #dct_decoded_image_file = "dct_" + original_image_file
        #cv2.imwrite(dct_decoded_image_file,sImg)
        return ''
      
    """Helper function to 'stitch' new image back together"""
    def chunks(self, l, n):
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
                    asc = b
                encoded.putpixel((col, row), (r, g , asc))
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
                    length = b
                elif index <= length:
                    msg += chr(b)
                index += 1
        lsb_decoded_image_file = "lsb_" + original_image_file
        #img.save(lsb_decoded_image_file)
        ##print("Decoded image was saved!")
        return msg

class Compare():
    def correlation(self, img1, img2):
        return signal.correlate2d (img1, img2)
    def meanSquareError(self, img1, img2):
        error = np.sum((img1.astype('float') - img2.astype('float')) ** 2)
        error /= float(img1.shape[0] * img1.shape[1]);
        return error
    def psnr(self, img1, img2):
        mse = self.meanSquareError(img1,img2)
        if mse == 0:
            return 100
        PIXEL_MAX = 255.0
        return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))



#driver part :
#deleting previous folders :
if os.path.exists("Encoded_image/"):
    shutil.rmtree("Encoded_image/")
if os.path.exists("Decoded_output/"):
    shutil.rmtree("Decoded_output/")
if os.path.exists("Comparison_result/"):
    shutil.rmtree("Comparison_result/")
#creating new folders :
os.makedirs("Encoded_image/")
os.makedirs("Decoded_output/")
os.makedirs("Comparison_result/")
original_image_file = ""    # to make the file name global variable
lsb_encoded_image_file = ""
dct_encoded_image_file = ""
dwt_encoded_image_file = ""

while True:
    m = input("To encode press '1', to decode press '2', to compare press '3', press any other button to close: ")

    if m == "1":
        os.chdir("Original_image/")
        original_image_file = input("Enter the name of the file with extension : ")
        lsb_img = Image.open(original_image_file)
        dct_img = cv2.imread(original_image_file, cv2.IMREAD_UNCHANGED)
#        dwt_img = cv2.imread(original_image_file, cv2.IMREAD_UNCHANGED)
        print("Description : ",lsb_img,"\nMode : ", lsb_img.mode)
        secret_msg = input("Enter the message you want to hide: ")
        print("The message length is: ",len(secret_msg))
        os.chdir("..")
        os.chdir("Encoded_image/")
        lsb_img_encoded = LSB().encode_image(lsb_img, secret_msg)
        dct_img_encoded = DCT().encode_image(dct_img, secret_msg)
#        dwt_img_encoded = DWT().encode_image(dwt_img, secret_msg)
        lsb_encoded_image_file = "lsb_" + original_image_file
        lsb_img_encoded.save(lsb_encoded_image_file)
        dct_encoded_image_file = "dct_" + original_image_file
        cv2.imwrite(dct_encoded_image_file,dct_img_encoded)
#        dwt_encoded_image_file = "dwt_" + original_image_file
#        cv2.imwrite(dwt_encoded_image_file,dwt_img_encoded) # saving the image with the hidden text
        print("Encoded images were saved!")
        os.chdir("..")

    elif m == "2":
        os.chdir("Encoded_image/")
        lsb_img = Image.open(lsb_encoded_image_file)
        dct_img = cv2.imread(dct_encoded_image_file, cv2.IMREAD_UNCHANGED)
#        dwt_img = cv2.imread(dwt_encoded_image_file, cv2.IMREAD_UNCHANGED)
        os.chdir("..") #going back to parent directory
        os.chdir("Decoded_output/")
        lsb_hidden_text = LSB().decode_image(lsb_img)
        dct_hidden_text = DCT().decode_image(dct_img) 
#        dwt_hidden_text = DWT().decode_image(dwt_img) 
        file = open("lsb_hidden_text.txt","w")
        file.write(lsb_hidden_text) # saving hidden text as text file
        file.close()
        file = open("dct_hidden_text.txt","w")
        file.write(dct_hidden_text) # saving hidden text as text file
        file.close()
#        file = open("dwt_hidden_text.txt","w")
#        file.write(dwt_hidden_text) # saving hidden text as text file
#        file.close()
        print("Hidden texts were saved as text file!")
        os.chdir("..")
    elif m == "3":
        #comparison calls
        os.chdir("Original_image/")
        original = cv2.imread(original_image_file)
        os.chdir("..")
        os.chdir("Encoded_image/")
        lsbEncoded = cv2.imread(lsb_encoded_image_file)
        dctEncoded = cv2.imread(dct_encoded_image_file)
#        dwtEncoded = cv2.imread(dwt_encoded_image_file)
        original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        lsb_encoded_img = cv2.cvtColor(lsbEncoded, cv2.COLOR_BGR2RGB)
        dct_encoded_img = cv2.cvtColor(dctEncoded, cv2.COLOR_BGR2RGB)
#        dwt_encoded_img = cv2.cvtColor(dwtEncoded, cv2.COLOR_BGR2RGB)
        os.chdir("..")
        os.chdir("Comparison_result/")

        book = xlwt.Workbook()
        sheet1=book.add_sheet("Sheet 1")
        style_string = "font: bold on , color red; borders: bottom dashed"
        style = xlwt.easyxf(style_string)
        sheet1.write(0, 0, "Original vs", style=style)
        sheet1.write(0, 1, "MSE", style=style)
        sheet1.write(0, 2, "PSNR", style=style)
        sheet1.write(1, 0, "LSB")
        sheet1.write(1, 1, Compare().meanSquareError(original, lsb_encoded_img))
        sheet1.write(1, 2, Compare().psnr(original, lsb_encoded_img))
        sheet1.write(2, 0, "DCT")
        sheet1.write(2, 1, Compare().meanSquareError(original, dct_encoded_img))
        sheet1.write(2, 2, Compare().psnr(original, dct_encoded_img))
        sheet1.write(3, 0, "DWT")
#        sheet1.write(3, 1, Compare().meanSquareError(original, dwt_encoded_img))
#        sheet1.write(3, 2, Compare().psnr(original, dwt_encoded_img))

        book.save("Comparison.xls")
        print("Comparison Results were saved as xls file!")
        os.chdir("..")
    else:
        print("Closed!")
        break
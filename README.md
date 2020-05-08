# Steganography

**_Hide text messages in images_**

## Description

(**TL;DR :** _This program can hide a text message in an image, which can only
be decoded by this program only._)

Steganography can be divided into two types:

1. Spatial domain transformation:
   - **LSB** (Least Significant Bit) transformation.
2. Frequency domain transformation:
   - **DCT** (Discrete Cosine Transformation).
   - **DWT** (Discrete Wavelet Transformation).

If you want to know about the transformations, and how they are performed, check
out our [Project Paper](Detailed_theory_paper/steganography.pdf).

## Usage:

Make sure you have **pip** installed!

Go to your terminal. And run the following command to install dependencies:

```console
pip install -r requirements.txt
```

Now to run the program, run the following command:

```console
python watermarking.py
```

It will create 3 new folders
`(it will overwrite if those folders previously exixsted)` named
**_"Encoded_image"_** , **_"Decoded_output"_** and **_"Comparison_result"_**.

### Encoding:

`Pressing 1` will ask you which carrier image you want to use, remember to enter
full name of any of the file.

```
lenna.png
pepper.png
babylon.png
```

**_You can also use any image you want as carrier, but make sure to add them
inside the folder named "Original_image"_**

Now this will ask you to enter the text which you want to hide inside that
image.
Then it will encode your text in the image and save it inside the folder named
**_"Encoded_image"_**.

### Decoding:

`Pressing 2` will create decoded `text file` from the image in the folder named
**_"Decoded_output"_** which contains the text you entered earlier.

### Comparison:

`Pressing 3` will create a `spreadsheet file` with the details of **MSE** (Mean
Square Error) and **PSNR** (Peak Signal to Noise Ratio) in the folder named
**_"Comparison_result"_**.

<img src="Doc/terminal.JPG">

##

## Now let's see what changed visually:

### Lenna:

<img src="Doc/original/lenna.png" width="250">
<img src="Doc/lsb_encoded/lsb_lenna.png" width="250">
<img src="Doc/dct_encoded/dct_lenna.png" width="250">

### Here is the comparison:

<img src="Doc/comparison_images/lenna_comparison.gif">

### Pepper:

<img src="Doc/original/pepper.png" width="250">
<img src="Doc/lsb_encoded/lsb_pepper.png" width="250">
<img src="Doc/dct_encoded/dct_pepper.png" width="250">

### Here is the comparison:

<img src="Doc/comparison_images/pepper_comparison.gif">

### Babylon:

<img src="Doc/original/babylon.png" width="250">
<img src="Doc/lsb_encoded/lsb_babylon.png" width="250">
<img src="Doc/dct_encoded/dct_babylon.png" width="250">

### Here is the comparison:

<img src="Doc/comparison_images/babylon_comparison.gif">

## Known issues:

```
DCT:
  You can see blue tint on the DCT encoded images.
  This code is now working on the blue channel of the images.
  If I specify any other RGB channel, like red or green, It gets tinted with that exact channel colour.
DWT:
  DWT doesn't work in this code for now.
```

## To-do:

**_Add implementation of DWT in this program._**

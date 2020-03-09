# Hop-Filed Neural Network

#### graphical\_test\_network.py
	- Accepts a training data file and a testing pattern.
	- Training file contains the pixels of all the inputs line by line.
	- Testing file contains one line containing the pixel values.

#### collect\_data.py
	- Acceps a file name to write data into.
	- You'll be presented with a 10x10 grid where you can draw patterns.
	- Press 'q' after you are done with drawing.
	- Now the pixel values will be written into the file in row major order.

#### represent\_network.py
	- Accepts a file name which contains the pixel values and then plots on a 10x10 grid.
	- Just to see which pattern are we exactly using.

#### test\_pattern.py
	- Works similar to graphical_test_network.py but without GUI.

* ## How to run?
### Collect Training Data
- Run collect\_data.py with a filename as argument.
	> Eg: ./collect\_data.py letter_A.txt
- Now draw the desired pattern on the provided 10x10 grid.
- When you are done with drawing, press 'q'.
- This way, collect as much data as you want.

### Combine data
- Put all the data in a single file.
	> Eg: cat letter_A.txt letter_B.txt letter_C.txt >> training\_data.txt

### Collect Distorted Data
- Run collect\_data.py file with a filename as argument.
	> Eg: ./collect\_data.py distorted_letter_A.txt
- Now draw desired distorted pattern and press 'q' when you are done.

### Test the Network
- run graphical\_test\_network.py with training\_data\_file and testing\_data\_file
	> Eg: ./graphical\_test\_network.py training\_data.txt distorted\_A.txt


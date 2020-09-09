import pandas as pd
import cv2 as cv

filename = "truth.csv"
truth = pd.read_csv(filename)

counter = 0
for index, row in truth.iterrows():
    
    if counter % 2 == 0:
        text_x = 10
    else:
        text_x = 1024 - 170

    text_y = 1024 - 100
    counter += 1

    image = row['image']
    background = row['background']
    class_ = row['class']
    if class_ != 'NORM':
        severity = row['severity']
        x0 = int(row['x0'])
        y0 = 1024 - int(row['y0'])
        radius = int(row['radius'])
    else:
        severity = ""

    filepath = image + ".pgm"
    img = cv.imread(filepath, 0)

    
    if class_ != 'NORM':
        cv.circle(img, (x0, y0), radius, (0, 0, 255), thickness=1, lineType=8, shift=0)

    font = cv.FONT_HERSHEY_SIMPLEX 
    cv.putText(img, " ".join([background, class_, severity]), (text_x, text_y), font, 1, (255, 255, 255), 2, cv.LINE_AA) 
    cv.imwrite("output/" + filepath, img)


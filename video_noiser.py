import random
import cv2
import numpy as np

# Load the video
cap = cv2.VideoCapture('videos\sport.mp4')

# Get the width and height of the frames
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object to save the output video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_video.mp4', fourcc, 20.0, (width, height))

# Number of rows and columns
rows, cols = 5, 5

# Calculate the size of each cell
cell_width = width // cols
cell_height = height // rows

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Loop through the matrix
    for i in range(rows):
        for j in range(cols):
            # You can specify which cells to make black, e.g., (i, j) == (2, 2) for the center cell
            if random.randint(0,1) == 1:  # Example cells to make black
                start_x = j * cell_width
                start_y = i * cell_height
                end_x = start_x + cell_width
                end_y = start_y + cell_height
                frame[start_y:end_y, start_x:end_x] = 0  # Make the cell black

    # Write the frame to the output video
    out.write(frame)

    # Display the frame for debugging (optional)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything when done
cap.release()
out.release()
cv2.destroyAllWindows()

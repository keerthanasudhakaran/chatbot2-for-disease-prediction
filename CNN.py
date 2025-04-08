
classes = [
    'Cyst', 'Kidney Stone','Medullary Sponge Kidney','Tumor'
]

class_mapping = {i: classes[i] for i in range(len(classes))}

print(class_mapping)


import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Load the saved model
model = load_model("C:/Users/ajith/Downloads/Kidney.h5")

# Function to preprocess an image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(256, 256))  # Assuming input size of (256, 256)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict(img_path):
    # Path to the image
    #img_path = 'C:/Users/ajith/Downloads/Dataset/Tumor/Tumor- (981).jpg'

    # Preprocess the image
    preprocessed_img = preprocess_image(img_path)

    # Make predictions
    predictions = model.predict(preprocessed_img)

    # Decode the predictions manually
    predicted_class_index = np.argmax(predictions[0])  # Get the index of the maximum probability
    predicted_class = f"Class{predicted_class_index + 1}"  # Add 1 to convert from 0-based to 1-based indexing

    # Display the image
    img = image.load_img(img_path)
    plt.imshow(img)
    plt.axis('off')
    plt.show()

    # Print the predicted class
    print("The uploaded CT scan image suggests the presence of ", class_mapping[predicted_class_index])
    return class_mapping[predicted_class_index]


#predict("C:/Users/ajith/Downloads/Dataset/Tumor/testtumour.jpg")





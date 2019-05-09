# 3d_printing_defect_detection

## Goal
Build a deep-learning algorithm for detecting defective layers of a 3D-printed detail. 

<table>
  <thead>
    <tr>
      <td>Good layer</td>
      <td>Defective layer</td>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><img src="./images/training.jpg" alt="Good layer image"></td>
      <td><img src="./images/defective.jpg" alt="Defective layer image"></td>
    </tr>
  </tbody>
</table>

## Data overview
The data was available for 3 printing machines, however only one machine was used for building the MVP and testing the approach. Major data properties:

* 44763 images of printed layers, both good and defective
* No labeleing for the images - prior to the MVP it was unknown which layers should be considered defective
* Layers are not consistent - specific "groups" of laeyrs could be defined based on the similar look

<table>
  <thead>
    <tr>
      <td>Group 1</td>
      <td>Group 2</td>
      <td>...</td>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><img src="./images/group_1.jpg" alt="Group 1 example"></td>
      <td><img src="./images/group_2.jpg" alt="Group 2 example"></td>
      <td>...</td>
    </tr>
  </tbody>
</table>

* Prior to the training the manual expection of part of the images was done, to identify at least some defective images for validation purposes
* Final training set: 43260 images
* Test set:
  - Holdout: 2277 images
  - Defective: 126 images
  
## Solution
### Neural network
The core of the solution is a Convolutionla Autoencoder with some logic applied above it for anomaly scoring.
 
Neural network architecute (drawn using <a href="http://alexlenail.me/NN-SVG/LeNet.html">this</a> tool):
<img src="./application/static/index/cnn_architecture.png" alt="NN architecure">

Same in code:
```python
#Random seed
np.random.seed(42)

# Network architecture
input_image_cnn = Input(shape=(256, 256, 1))

# Encoding
encoded_cnn = Conv2D(32, (5, 5), activation='relu', padding='same')(input_image_cnn)
encoded_cnn = MaxPooling2D((2, 2), padding='same')(encoded_cnn)

encoded_cnn = Conv2D(32, (5, 5), activation='relu', padding='same')(encoded_cnn)
encoded_cnn = MaxPooling2D((2, 2), padding='same')(encoded_cnn)

encoded_cnn = Conv2D(16, (5, 5), activation='relu', padding='same')(encoded_cnn)
encoded_cnn = MaxPooling2D((2, 2), padding='same')(encoded_cnn)


# Decoding
decoded_cnn = Conv2D(16, (5, 5), activation='relu', padding='same')(encoded_cnn)
decoded_cnn = UpSampling2D((2, 2))(decoded_cnn)

decoded_cnn = Conv2D(32, (5, 5), activation='relu', padding='same')(decoded_cnn)
decoded_cnn = UpSampling2D((2, 2))(decoded_cnn)

decoded_cnn = Conv2D(32, (5, 5), activation='relu', padding='same')(decoded_cnn)
decoded_cnn = UpSampling2D((2, 2))(decoded_cnn)

decoded_cnn = Conv2D(1, (5, 5), activation='sigmoid', padding='same')(decoded_cnn)
```

### Anomaly logic

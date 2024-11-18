# Music Sequence Prediction using LSTMs

*Written by Dr. Michael Lomuscio*

This Google Colab notebook demonstrates a basic implementation of a music sequence prediction model using Long Short-Term Memory (LSTM) networks, a type of Recurrent Neural Network (RNN). The primary focus of this project is to predict the next musical note in a sequence based on the preceding notes, simulating how music might be composed or predicted based on prior patterns. Below, we'll provide a brief introduction to LSTMs and how they are utilized in this project.

## Key Concepts and Models Used

### 1. LSTM Networks
LSTMs are a type of Recurrent Neural Network designed to recognize patterns in sequences of data. They are particularly useful for time-series data like music, where past notes have an influence on future notes. In this code, LSTMs are employed to learn the relationships between notes and predict the next note in a sequence. LSTMs are ideal for this task because they have memory cells that can maintain information over longer time intervals, making them well-suited for dealing with sequential data where the order of elements matters.

### 2. Music Note Mapping and Sequence Creation
The model first converts a set of musical notes (C, D, E, F, G, A, B) into numeric representations, which are more suitable for machine learning algorithms. This numerical encoding allows the model to process the notes as numerical data, which is essential for training the LSTM. In this notebook, a sequence of musical notes is defined, and input-output pairs are generated to train the model.

### 3. Model Structure
The model uses a Sequential API from TensorFlow's Keras library to create a simple LSTM-based neural network. It has two main layers:
- **LSTM Layer:** This layer processes the sequence of notes and captures the temporal dependencies among them.
- **Dense Layer:** The Dense layer with a softmax activation outputs a probability distribution over the possible notes, allowing the model to predict the next note.

### 4. Training Process
The model is trained to predict the next note in a sequence based on a sliding window of previous notes. The training data is constructed by slicing the original sequence into smaller overlapping subsequences. For example, given the sequence [C, D, E, F, G, A, B, C, D, E], the model learns using smaller chunks like [C, D, E] to predict F, [D, E, F] to predict G, and so on.

## How to Use This Code

1. **Dependencies**  
   Make sure you have TensorFlow and NumPy installed. You can install them in Google Colab using the following commands:
   ```python
   !pip install tensorflow
   !pip install numpy
## Custom Input Sequence Prediction Using LSTM

This section of the Google Colab notebook extends the functionality of our LSTM-based music prediction model by allowing us to make predictions on a custom input sequence of notes. The idea is to feed a sequence of notes into the pre-trained model and have it predict the next note in the sequence. This is particularly useful for experimenting with different musical sequences, understanding how the model interprets them, and exploring the potential of generating new music based on user-defined inputs.

### Overview of Key Steps and Ideas

**1. Define the Custom Sequence of Notes**  
- The code begins by allowing you to define a custom sequence of notes that you wish to use for prediction. For instance, you can replace the sequence `['E', 'A', 'A']` with any set of notes you are interested in. The ability to specify your own sequence offers flexibility in testing the model's ability to predict different musical contexts.

**2. Data Validation and Conversion**  
- The next step ensures that all notes in the custom sequence are valid. Specifically, the code checks if each note is part of the predefined set of notes (`the_notes`). If any note is not recognized, the code raises an error to prevent incorrect input from being processed.
- Once validated, the custom sequence is converted to its integer representation using the mapping dictionary (`note_to_int`). This conversion is necessary because the LSTM model expects numerical input rather than text-based note labels.

**3. Prepare Input Data for the Model**  
- The custom sequence, now represented by integers, is converted into a NumPy array. To feed it into the model, the sequence is reshaped to meet the LSTM input requirements, which expect input in the shape of `[samples, time steps, features]`. Here, the input is reshaped to have one sample, with the length of the custom sequence representing the time steps, and each time step having one feature.
- Additionally, the input data is normalized by dividing it by the number of notes (`float(len(the_notes))`). Normalization helps improve the model's learning efficiency by ensuring that all input values fall within the range [0, 1]. This step is crucial for maintaining consistency between the format of the training data and the custom input data.

**4. Making Predictions with the Model**  
- After preparing the input, the code uses the pre-trained LSTM model to predict the next note in the sequence. The model outputs a probability distribution over all possible notes, indicating the likelihood of each note being the next in the sequence.
- The `np.argmax()` function is then used to identify the note with the highest probability, which is considered the model's prediction for the next note.
- Finally, the predicted note index is converted back to its original note label using the `int_to_note` mapping dictionary. This allows us to display the result in a human-readable format, showing both the input sequence and the predicted next note.

### How to Use This Code

1. **Modify the Custom Sequence**  
   - You can customize the sequence of notes by changing the `custom_sequence` variable. For example, you might choose `['C', 'G', 'E']` or any other combination of notes to see how the model continues the sequence.

2. **Run the Code**  
   - Ensure that the model has already been trained before running this custom prediction code. Otherwise, the model won't be able to generate meaningful predictions.
   - Execute each step in order to convert the notes, prepare the input, and make a prediction.

3. **Interpreting the Output**  
   - The code will print the custom input sequence you provided, as well as the predicted next note. The prediction is based on the patterns learned by the LSTM during training, and it is interesting to see whether the prediction aligns with typical musical expectations.

### Practical Applications and Extensions

- **Music Composition:** This functionality can be used for music composition by allowing users to iteratively generate new notes, potentially creating entire pieces of music by repeatedly feeding predictions back into the model.
- **Experimenting with Different Sequences:** You can explore how different input sequences lead to different predictions, helping you understand the types of musical patterns the model has learned.
- **Interactive Music Generation:** By wrapping this process in a user interface, you could build an interactive music generation tool where users can input sequences and hear the model's predicted continuations in real time.

This extension of the LSTM model to custom input sequences provides a powerful tool for anyone interested in understanding or creating music using deep learning. It highlights the potential of AI in generating creative outputs that can augment human creativity.

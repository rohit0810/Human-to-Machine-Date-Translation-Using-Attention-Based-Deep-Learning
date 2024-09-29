# Human-to-Machine-Date-Translation-Using-Attention-Based-Deep-Learning
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Seq2Seq Date Converter with Attention</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 20px;
            max-width: 800px;
        }
        h1, h2, h3 {
            color: #333333;
        }
        code {
            background-color: #f4f4f4;
            padding: 2px 4px;
            border-radius: 4px;
        }
        pre {
            background-color: #f4f4f4;
            padding: 10px;
            border-radius: 4px;
            overflow-x: auto;
        }
        a {
            color: #1a0dab;
            text-decoration: none;
        }
        a:hover {
            text-decoration: underline;
        }
        .badge {
            display: inline-block;
            padding: 0.25em 0.4em;
            font-size: 75%;
            font-weight: 700;
            line-height: 1;
            text-align: center;
            white-space: nowrap;
            vertical-align: baseline;
            border-radius: 0.25rem;
            background-color: #28a745;
            color: white;
        }
    </style>
</head>
<body>
    <h1>Seq2Seq Date Converter with Attention</h1>
    
    <p>
        <span class="badge">96% Accuracy</span>
    </p>
    
    <h2>Project Overview</h2>
    <p>
        This project implements a Sequence-to-Sequence (Seq2Seq) model with Attention mechanism using PyTorch to convert human-readable date formats into machine-readable formats. The model achieves an impressive <strong>96% accuracy</strong> on the validation dataset.
    </p>
    
    <h2>Features</h2>
    <ul>
        <li>Seq2Seq architecture with Encoder and Decoder components.</li>
        <li>Attention mechanism to enhance model performance.</li>
        <li>Bidirectional LSTM in the encoder for better context understanding.</li>
        <li>Teacher forcing during training for efficient learning.</li>
        <li>Inference capabilities with attention visualization.</li>
        <li>Handles variable-length input and output sequences with padding.</li>
    </ul>
    
    <h2>Installation</h2>
    <p>Follow the steps below to set up the project:</p>
    <ol>
        <li>Clone the repository:</li>
        <pre><code>git clone https://github.com/yourusername/seq2seq-date-converter.git</code></pre>
        
        <li>Navigate to the project directory:</li>
        <pre><code>cd seq2seq-date-converter</code></pre>
        
        <li>Create and activate a virtual environment (optional but recommended):</li>
        <pre><code>python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate</code></pre>
        
        <li>Install the required packages:</li>
        <pre><code>pip install -r requirements.txt</code></pre>
    </ol>
    
    <h2>Usage</h2>
    <h3>Training the Model</h3>
    <p>
        Ensure that your training and validation datasets are placed in the appropriate directories and update the file paths in the script if necessary.
    </p>
    <pre><code>python train.py</code></pre>
    
    <h3>Performing Inference</h3>
    <p>
        After training, you can perform inference on new data using the following command:
    </p>
    <pre><code>python inference.py</code></pre>
    
    <h3>Example Inference</h3>
    <p>
        <strong>Input:</strong> <code>23 January 01</code><br>
        <strong>Predicted Output:</strong> <code>2023-01-23</code>
    </p>
    
    <h2>Results</h2>
    <p>The model was trained for <strong>6 epochs</strong> with the following configuration:</p>
    <ul>
        <li>Input Sequence Length (Tx): 30</li>
        <li>Output Sequence Length (Ty): 12</li>
        <li>Batch Size: 64</li>
        <li>Learning Rate: 0.001</li>
        <li>Weight Decay: 1e-4</li>
        <li>Hidden Dimension: 128</li>
        <li>Decoder Hidden Dimension: 128</li>
        <li>Attention Dimension: 64</li>
        <li>Bidirectional Encoder: False</li>
        <li>Dropout: 0.5</li>
    </ul>
    <p>
        The model achieved a <strong>96% exact match accuracy</strong> on the validation set, demonstrating its effectiveness in accurately converting date formats.
    </p>
    
    <h2>Attention Visualization</h2>
    <p>
        The model includes functionality to visualize attention weights, providing insights into how the model focuses on different parts of the input sequence during decoding.
    </p>
    <p>
        <img src="AttentionMap.png" alt="Attention Map" style="max-width:100%;">
    </p>
    
    <h2>Project Structure</h2>
    <pre><code>
seq2seq-date-converter/
│
├── data/
│   ├── Assignment2_train.txt
│   ├── Assignment2_validation.txt
│   └── Assignment2_Test.txt
│
├── models/
│   └── seq2seq_date_converter_rohit.pth
│
├── outputs/
│   └── answer_rohit.xlsx
│
├── src/
│   ├── train.py
│   ├── inference.py
│   └── utils.py
│
├── AttentionMap.png
├── requirements.txt
└── README.html
    </code></pre>
    
    <h2>Dependencies</h2>
    <ul>
        <li>Python 3.6+</li>
        <li>PyTorch</li>
        <li>NumPy</li>
        <li>Matplotlib</li>
        <li>Pandas</li>
        <li>Babel</li>
        <li>TQDM</li>
        <li>OpenPyXL</li>
    </ul>
    
    <h2>Contributing</h2>
    <p>
        Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.
    </p>
    
    <h2>License</h2>
    <p>
        This project is licensed under the MIT License. See the <a href="LICENSE">LICENSE</a> file for details.
    </p>
    
    <h2>Contact</h2>
    <p>
        For any questions or suggestions, please contact <a href="mailto:your.email@example.com">your.email@example.com</a>.
    </p>
    
    <h2>Acknowledgments</h2>
    <ul>
        <li>Inspired by the Seq2Seq models in natural language processing.</li>
        <li>Thanks to the PyTorch community for their excellent resources and support.</li>
    </ul>
</body>
</html>

---

# Email Classifier

This project is an **Email Classifier** that categorizes emails into different classes such as **Spam** and **Not Spam (Ham)**. The classifier is built using Python and employs the **Naive Bayes algorithm** for classification. The primary goal is to identify spam emails effectively and ensure the proper categorization of incoming messages.

---

## Features

- **Spam Detection**: Distinguishes between spam and ham emails.
- **Text Processing**: Utilizes techniques like **Bag-of-Words** and **TF-IDF Transformation**.
- **Machine Learning**: Implements a **Naive Bayes classifier** for efficient text classification.
- **Test Data**: Includes examples to test the classifier's performance.

---

## Installation and Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/ibrahimfadu/Email-classifier.git
   cd Email-classifier
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Jupyter Notebook:
   ```bash
   jupyter notebook spam-classifier.ipynb
   ```

---

## How It Works

1. **Dataset**: The dataset consists of labeled emails classified as `spam` or `ham`. You can use the provided sample dataset or replace it with your own.
2. **Text Preprocessing**:
   - Tokenization
   - Conversion to Bag-of-Words representation
   - TF-IDF transformation
3. **Model Training**:
   - Trains a **Naive Bayes Classifier** using the processed text data.
4. **Evaluation**:
   - Provides performance metrics like **Accuracy**, **Precision**, **Recall**, and **F1-Score**.
5. **Prediction**:
   - Test the classifier with new email samples to predict their categories.

---

## Example Usage

After training, you can test the classifier with new emails. For instance:

```python
test_emails = [
    'You have won a free trip to Paris!',
    'The team meeting has been postponed to 3 PM.'
]
predictions = pipeline.predict(test_emails)
print(predictions)
```

**Output**:
```
['spam', 'ham']
```

---

## Dependencies

- Python 3.x
- Pandas
- Scikit-learn
- Jupyter Notebook
- Matplotlib (optional, for visualizations)

Install all dependencies using:
```bash
pip install -r requirements.txt
```

---

## Project Files

- **spam-classifier.ipynb**: The main Jupyter Notebook for building and testing the classifier.
- **requirements.txt**: List of dependencies.
- **README.md**: Project documentation (this file).

---

## Future Improvements

- Add support for more email categories (e.g., Promotions, Social).
- Use advanced models like **Logistic Regression**, **SVM**, or **Transformers** for better accuracy.
- Incorporate real-world datasets like the [Enron Email Dataset](https://www.cs.cmu.edu/~enron/).

---
## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

## Acknowledgments

Special thanks to the creators of:
- [Scikit-learn](https://scikit-learn.org/)
- [Jupyter](https://jupyter.org/)

---

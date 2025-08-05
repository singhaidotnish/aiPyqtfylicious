import sys
import openai
from openai import OpenAI
import os
from dotenv import load_dotenv
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QTextEdit,
    QLineEdit, QPushButton
)

# Load from .env
load_dotenv()
client = OpenAI()  # 👈 this was missing in your version
# Set the API key
openai.api_key = os.getenv("OPENAI_API_KEY")

class AIChatBot(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ask AI")
        self.setGeometry(100, 100, 500, 300)

        layout = QVBoxLayout()

        self.label = QLabel("Enter your question:")
        self.input_field = QLineEdit()
        self.answer_box = QTextEdit()
        self.answer_box.setReadOnly(True)
        self.ask_button = QPushButton("Ask AI")
        self.ask_button.clicked.connect(self.ask_ai)

        layout.addWidget(self.label)
        layout.addWidget(self.input_field)
        layout.addWidget(self.ask_button)
        layout.addWidget(self.answer_box)

        self.setLayout(layout)

    def ask_ai(self):
        question = self.input_field.text()
        if not question.strip():
            self.answer_box.setText("Please enter a question.")
            return

        self.answer_box.setText("Thinking...")

        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": question}]
            )
            answer = response.choices[0].message.content.strip()
        except Exception as e:
            answer = f"Error: {e}"

        self.answer_box.setText(answer)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = AIChatBot()
    window.show()
    sys.exit(app.exec_())

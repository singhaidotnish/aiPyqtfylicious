import sys
import openai
from openai import OpenAI
import os
from dotenv import load_dotenv
import datetime

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QTextEdit,
    QLineEdit, QPushButton
)

# Load from .env
load_dotenv()

class AIChatBot(QWidget):
    def __init__(self):
        super().__init__()

        # 👇 FIX: create client here
        self.client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama"
        )
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
        question = self.input_field.text().strip()
        if not question:
            return

        # Inject system time if the question is about time
        if "time" in question.lower():
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            answer = f"The current system time is: {now}"
            self.answer_box.setText(answer)
            return

        try:
            response = self.client.chat.completions.create(
                model="llama3.1:8b",
                messages=[{"role": "user", "content": question}],
            )
            answer = response.choices[0].message.content
            self.answer_box.setText(answer)
        except Exception as e:
            self.answer_box.setText(f"Error: {str(e)}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = AIChatBot()
    window.show()
    sys.exit(app.exec_())

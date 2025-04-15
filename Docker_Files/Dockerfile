FROM python:3.10-slim

WORKDIR /app

# Copy requirements and install dependencies

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy the project code (including app.py)

COPY . .

# Expose Streamlit port

EXPOSE 8501

# Set environment variable for headless mode

ENV STREAMLIT_SERVER_HEADLESS=true

# Command to run the Streamlit app

CMD ["streamlit", "run", "app.py"]
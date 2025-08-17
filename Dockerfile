Use a lightweight official Python image.

FROM python:3.11-slim

Set the working directory inside the container.

WORKDIR /app

Copy the requirements file and install dependencies.

The --no-cache-dir flag reduces image size.

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

Copy the Streamlit application file.

COPY app.py ./

Expose the port that Streamlit runs on.

EXPOSE 8501

Command to run the Streamlit application.

The --server.port and --server.address flags are important for Docker.

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

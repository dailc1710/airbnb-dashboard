# Project Overview

This project is an **Airbnb Data Analytics Dashboard** built using Streamlit. The application allows users to explore Airbnb listing data, view interactive visualizations, and gain insights from the data. The dashboard is designed to be a modern analytics tool, similar in look and feel to PowerBI.

The project is structured as a multi-page Streamlit application with a clear separation of concerns. The `core` directory contains the main business logic, including data loading and preprocessing, while the `pages` directory holds the UI components for each page of the dashboard. User authentication is required to access the dashboard.

**Key Technologies:**

*   **Backend:** Python
*   **Frontend:** Streamlit
*   **Data Science Libraries:** pandas, numpy, scikit-learn, plotly

# Building and Running

To run this project, you need to have Python and the required packages installed.

**1. Install Dependencies:**

```bash
pip install -r requirements.txt
```

**2. Run the Application:**

```bash
streamlit run app.py
```

This will start a local web server, and you can access the application in your browser at `http://localhost:8501`.

**3. Testing:**

There are no explicit test commands or frameworks configured in this project.

# Development Conventions

*   **Code Style:** The code follows the PEP 8 style guide for Python.
*   **File Structure:** The project is organized into modules, with a clear separation between the core logic (`core` directory) and the UI components (`pages` directory).
*   **Authentication:** The application uses a session-based authentication system to control access to the dashboard.
*   **Data Handling:** Data is loaded from a CSV file (`data/Airbnb_Data_cleaned.csv`). If the file is not found, a sample dataset is generated. Data preprocessing and cleaning are performed before the data is used for analysis and visualization.
*   **Internationalization (i18n):** The application supports multiple languages, with language selection available in the sidebar.
*   **Styling:** The application uses custom CSS to style the UI components.

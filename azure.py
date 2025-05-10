import os
import streamlit as st
import pandas as pd
import pyodbc
import requests
from io import StringIO
import json
import re
from datetime import datetime, timedelta
import time
from fpdf import FPDF
import plotly.express as px

# Ensure necessary Langchain/Groq imports are present
try:
    from langchain_groq import ChatGroq
    from langchain_core.prompts import PromptTemplate
    from langchain_core.messages import HumanMessage, AIMessage
    from langchain_core.output_parsers import StrOutputParser

    LANGCHAIN_AVAILABLE = True
except ImportError:
    st.warning(
        "Langchain/Groq libraries not found. AI features will be disabled. Install with: pip install langchain langchain-groq fpdf plotly"
    )
    LANGCHAIN_AVAILABLE = False


    # Define dummy classes/functions if Langchain is not available to avoid NameErrors later
    class ChatGroq:
        def __init__(self, **kwargs):
            pass

        def invoke(self, messages):
            # Simulate AIMessage structure with content attribute
            dummy_response = lambda: None
            dummy_response.content = "AI features are disabled. Please install required packages."
            return dummy_response


    class PromptTemplate:
        def __init__(self):
            pass

        @staticmethod
        def from_template(text):
            # Store template text for potential inspection later if needed
            prompt_template = lambda: None  # Simple object to hold attributes
            prompt_template.template = text
            # Basic extraction of variables (might not be perfectly robust for complex templates)
            prompt_template.input_variables = re.findall(r"\{(\w+)\}", text)

            # Return a dummy object that can be invoked (returning formatted text)
            class DummyPrompt:
                def __init__(self, template, variables):
                    self.template = template
                    self.input_variables = variables

                def format(self, **kwargs):
                    formatted_text = self.template
                    for key, value in kwargs.items():
                        # Escape braces in values to prevent accidental formatting
                        escaped_value = str(value).replace("{", "{{").replace("}", "}}")
                        formatted_text = formatted_text.replace(f"{{{key}}}", escaped_value)
                    return formatted_text

            return DummyPrompt(text, prompt_template.input_variables)


    class HumanMessage:
        def __init__(self, content): self.content = content  # Basic init for dummy


    class AIMessage:
        def __init__(self, content): self.content = content  # Basic init for dummy

    class StrOutputParser:
        def __init__(self): pass  # Dummy init

        def invoke(self, input_data):
            # Assuming input_data is the AIMessage object or similar
            if hasattr(input_data, 'content'):
                return str(input_data.content)
            return str(input_data) # Fallback


# === Constants ===
# Azure SQL Database connection parameters for the *default* database used by the app
DB_SERVER = os.getenv("DB_SERVER", "agentdb123.database.windows.net")
DB_NAME = os.getenv("DB_NAME", "compliance_db")
DB_PORT = os.getenv("DB_PORT", 1433) # Port is often 1433 for Azure SQL


# === Authentication ===
def login():
    """Handles user login via sidebar."""
    st.sidebar.title("🔐 Login")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")

    # Default credentials if not found in secrets or environment variables
    app_user = os.getenv("APP_USERNAME", "admin")
    app_pass = os.getenv("APP_PASSWORD", "pass123")

    # Prefer secrets over environment variables if secrets are available
    secrets_available = hasattr(st, 'secrets')
    if secrets_available:
        try:
            app_user = st.secrets.get("APP_USERNAME", app_user)
            app_pass = st.secrets.get("APP_PASSWORD", app_pass)
        except Exception as e:
             st.sidebar.warning(f"Could not read APP login secrets: {e}. Using default or env vars.")


    if st.sidebar.button("Login"):
        if username == app_user and password == app_pass:
            st.session_state.logged_in = True
            st.rerun()
        else:
            st.sidebar.error("Invalid username or password")

# Initialize login state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# Enforce login
if not st.session_state.logged_in:
    login()
    # Provide guidance on default credentials and secrets/env vars
    try:
        secrets_or_env_set = (os.getenv("APP_USERNAME") or (hasattr(st, 'secrets') and st.secrets.get("APP_USERNAME")))
        if not secrets_or_env_set:
             st.sidebar.info("Default login: admin / pass123 (Set APP_USERNAME/APP_PASSWORD in secrets.toml or env vars)")
        else:
             st.sidebar.info("Using custom login from secrets/env vars.")
    except Exception:
         st.sidebar.info("Default login: admin / pass123 (Set APP_USERNAME/APP_PASSWORD in secrets.toml or env vars)")

    st.stop()  # Stop execution if not logged in

# === App Setup ===
st.set_page_config(page_title="Unified Banking Compliance Solution", layout="wide")


# === Azure SQL Database Connection Helper (for the *default* app DB) ===
@st.cache_resource(ttl="1h") # Cache connection for 1 hour
# Fix for get_db_connection() function

def get_db_connection():
    """
    Creates and returns a connection to the Azure SQL database using the credentials
    in st.secrets or environment variables, and global DB_SERVER/DB_NAME.
    Returns None if connection fails or credentials are not found.
    """
    db_username = None
    db_password = None
    use_entra = False
    entra_domain = None

    secrets_available = hasattr(st, 'secrets')

    # Prioritize Streamlit secrets
    if secrets_available:
        try:
            db_username = st.secrets.get("DB_USERNAME")
            db_password = st.secrets.get("DB_PASSWORD")
            use_entra_str = st.secrets.get("USE_ENTRA_AUTH", "false")
            use_entra = use_entra_str.lower() == "true"
            if use_entra:
                 entra_domain = st.secrets.get("ENTRA_DOMAIN")
                 if not entra_domain:
                      st.warning("USE_ENTRA_AUTH is true, but ENTRA_DOMAIN is missing in secrets.toml.")

        except Exception as e:
            st.warning(f"Could not read DB secrets: {e}. Trying environment variables.")

    # Fallback to Environment Variable
    if db_username is None or db_password is None:
        db_username = os.getenv("DB_USERNAME")
        db_password = os.getenv("DB_PASSWORD")
        if not use_entra: # Only check env var for Entra if not already set by secrets
            use_entra_str = os.getenv("USE_ENTRA_AUTH", "false")
            use_entra = use_entra_str.lower() == "true"
            if use_entra:
                 entra_domain = os.getenv("ENTRA_DOMAIN")
                 if not entra_domain:
                      st.warning("USE_ENTRA_AUTH env var is true, but ENTRA_DOMAIN env var is missing.")

    if not db_username or not db_password:
        st.error(
            "Database credentials (DB_USERNAME, DB_PASSWORD) not found in Streamlit secrets or environment variables."
        )
        st.info(
            "To connect to Azure SQL Database, ensure you have set:\n"
            "- `DB_USERNAME` and `DB_PASSWORD` in `.streamlit/secrets.toml` or as environment variables.\n"
            "- Optionally, `USE_ENTRA_AUTH=true` and `ENTRA_DOMAIN='yourdomain.onmicrosoft.com'` for Entra auth."
        )
        return None

    conn_str = ""
    try:
        if use_entra:
            if not entra_domain:
                 st.error("Microsoft Entra Authentication requires ENTRA_DOMAIN.")
                 return None
            conn_str = (
                f"DRIVER={{ODBC Driver 18 for SQL Server}};"  # Try newer driver first
                f"SERVER={DB_SERVER};" 
                f"DATABASE={DB_NAME};"
                f"Authentication=ActiveDirectoryPassword;"
                f"UID={db_username}@{entra_domain};"
                f"PWD={db_password};"
                f"Encrypt=yes;TrustServerCertificate=no;Connection Timeout=120;"
            )
            st.sidebar.caption("Attempting Entra Auth")
        else:
             # First attempt with numeric port in SERVER parameter
             conn_str = (
                f"DRIVER={{ODBC Driver 18 for SQL Server}};"  # Try newer driver first
                f"SERVER={DB_SERVER},{DB_PORT};" 
                f"DATABASE={DB_NAME};"
                f"UID={db_username};"
                f"PWD={db_password};"
                f"Encrypt=yes;TrustServerCertificate=no;Connection Timeout=120;"
             )
             st.sidebar.caption(f"Attempting SQL Auth to {DB_SERVER},{DB_PORT}")

        try:
            connection = pyodbc.connect(conn_str)
            st.sidebar.success("✅ Connected to default database.")
            return connection
        except pyodbc.Error as e:
            # If first attempt fails, try with older driver
            if "ODBC Driver 18 for SQL Server" in conn_str:
                st.sidebar.warning("Driver 18 failed, trying Driver 17...")
                conn_str = conn_str.replace("ODBC Driver 18 for SQL Server", "ODBC Driver 17 for SQL Server")
                connection = pyodbc.connect(conn_str)
                st.sidebar.success("✅ Connected to default database with ODBC Driver 17.")
                return connection
            # If still failing, try without port specification
            elif "," in conn_str and not use_entra:
                st.sidebar.warning("Connection with port failed, trying without port...")
                conn_str = (
                    f"DRIVER={{ODBC Driver 17 for SQL Server}};"
                    f"SERVER={DB_SERVER};" # Without port
                    f"DATABASE={DB_NAME};"
                    f"UID={db_username};"
                    f"PWD={db_password};"
                    f"Encrypt=yes;TrustServerCertificate=no;Connection Timeout=120;"
                )
                connection = pyodbc.connect(conn_str)
                st.sidebar.success("✅ Connected to default database without port specification.")
                return connection
            else:
                raise e  # Re-raise the exception if all connection attempts failed

    except pyodbc.Error as e:
        st.sidebar.error(f"Default Database Connection Error: {e}")
        # More specific error message based on error code
        if "08001" in str(e):
            st.sidebar.warning("Cannot reach the server. Check server name, firewall rules, and network connection.")
        elif "28000" in str(e):
            st.sidebar.warning("Login failed. Check username and password.")
        elif "42000" in str(e):
            st.sidebar.warning("Database access error. Check if the database exists and user has permission.")
        elif "01000" in str(e) and "TLS" in str(e):
            st.sidebar.warning("SSL/TLS error. Try setting TrustServerCertificate=yes in connection string.")
        else:
            st.sidebar.warning("Please check DB credentials, server address, database name, and firewall rules.")
        return None
    except Exception as e:
        st.sidebar.error(f"An unexpected error occurred during default DB connection: {e}")
        return None

# === Load LLM (Requires secrets.toml/env var and Langchain installation) ===
@st.cache_resource(show_spinner="Loading AI Assistant...")
def load_llm():
    """Loads the Groq LLM using API key from st.secrets or environment variables."""
    if not LANGCHAIN_AVAILABLE: return None

    api_key = None
    secrets_available = hasattr(st, 'secrets')

    # Prioritize Streamlit secrets
    if secrets_available:
        try:
            api_key = st.secrets.get("GROQ_API_KEY")
        except Exception as e:
             st.error(f"Error accessing GROQ_API_KEY secrets: {e}. Trying environment variable.")

    # Fallback to Environment Variable
    if api_key is None:
        api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        st.error("❗️ GROQ API Key not found.")
        st.info(
            "To use the AI features, please ensure:\n"
            "1. The file `.streamlit/secrets.toml` exists and contains `GROQ_API_KEY = \"YOUR_ACTUAL_GROQ_API_KEY\"` OR\n"
            "2. A `GROQ_API_KEY` environment variable is set.\n"
            "3. You have restarted the Streamlit app after setting."
        )
        return None

    try:
        # Use a smaller model if 70b is too slow or costly, e.g., "llama3-8b-8192"
        llm_instance = ChatGroq(temperature=0.2, model_name="llama3-70b-8192", api_key=api_key, request_timeout=120)
        st.sidebar.success("✅ AI Assistant Loaded!")
        return llm_instance
    except Exception as e:
        st.error(f"🚨 Failed to initialize Groq client: {e}")
        st.info("Please verify your GROQ_API_KEY value and ensure you have internet connectivity.")
        return None

llm = load_llm()


# === Database Setup ===
def init_db():
    """ Initializes the Azure SQL database and tables using the default connection."""
    conn = get_db_connection()
    if conn is None:
        st.error("Cannot initialize database: Default DB connection failed.")
        return False # Do not stop execution, allow app to run in disconnected mode

    try:
        with conn:
            cursor = conn.cursor()

            # Check if tables exist before creating them
            cursor.execute("""
                IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'accounts_data')
                CREATE TABLE accounts_data (
                    Account_ID NVARCHAR(255),
                    Account_Type NVARCHAR(255),
                    Last_Transaction_Date DATETIME2, -- Use DATETIME2 for better precision if needed
                    Account_Status NVARCHAR(255),
                    Email_Contact_Attempt NVARCHAR(255),
                    SMS_Contact_Attempt NVARCHAR(255),
                    Phone_Call_Attempt NVARCHAR(255),
                    KYC_Status NVARCHAR(255),
                    Branch NVARCHAR(255)
                )
            """)

            cursor.execute("""
                IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'dormant_flags')
                CREATE TABLE dormant_flags (
                    account_id NVARCHAR(255) PRIMARY KEY, -- Added PRIMARY KEY for clarity
                    flag_instruction NVARCHAR(MAX),
                    timestamp DATETIME2 DEFAULT CURRENT_TIMESTAMP -- Use DATETIME2
                )
            """)

            cursor.execute("""
                IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'dormant_ledger')
                CREATE TABLE dormant_ledger (
                    account_id NVARCHAR(255) PRIMARY KEY, -- Added PRIMARY KEY
                    classification NVARCHAR(255),
                    timestamp DATETIME2 DEFAULT CURRENT_TIMESTAMP -- Use DATETIME2
                )
            """)

            cursor.execute("""
                IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'insight_log')
                CREATE TABLE insight_log (
                    id INT IDENTITY(1,1) PRIMARY KEY,
                    timestamp DATETIME2 DEFAULT CURRENT_TIMESTAMP, -- Use DATETIME2 and default
                    observation NVARCHAR(MAX),
                    trend NVARCHAR(MAX),
                    insight NVARCHAR(MAX),
                    action NVARCHAR(MAX)
                )
            """)

            cursor.execute("""
                IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'sql_query_history')
                CREATE TABLE sql_query_history (
                    id INT IDENTITY(1,1) PRIMARY KEY,
                    natural_language_query NVARCHAR(MAX),
                    sql_query NVARCHAR(MAX),
                    timestamp DATETIME2 DEFAULT CURRENT_TIMESTAMP -- Use DATETIME2
                )
            """)

            conn.commit()
            st.sidebar.success("✅ Database schema initialized/verified.")
        return True
    except pyodbc.Error as e:
        st.error(f"Database Initialization Error: {e}")
        return False
    except Exception as e:
        st.error(f"An unexpected error occurred during DB initialization: {e}")
        return False


# Initialize database tables (attempt connection and creation)
db_initialized = init_db()


# === Helper Functions ===
# Updated parse_data function with improved error handling and data processing

@st.cache_data(show_spinner="Parsing data...")
def parse_data(file_input):
    """Parses data, standardizes column names, converts types, and stores original names."""
    df = None
    original_columns = []
    try:
        if isinstance(file_input, pd.DataFrame):
            st.sidebar.info("Processing data from DataFrame object...")
            df = file_input.copy()
            original_columns = list(df.columns)
        elif hasattr(file_input, 'name'):
            name = file_input.name.lower()
            st.sidebar.info(f"Processing file: {name}")
            if name.endswith('.csv'):
                df = pd.read_csv(file_input)
            elif name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_input, engine='openpyxl')
            elif name.endswith('.json'):
                df = pd.read_json(file_input)
            else:
                st.sidebar.error("Unsupported file format. Please use CSV, XLSX, or JSON.")
                return None
            if df is not None: original_columns = list(df.columns)
        elif isinstance(file_input, str):  # Handle URL fetched string data
            st.sidebar.info("Processing data from URL or text string...")
            df = pd.read_csv(StringIO(file_input))  # Assuming URL content is CSV
            if df is not None: original_columns = list(df.columns)
        else:
            st.sidebar.error(f"Invalid input type for parsing: {type(file_input)}")
            return None

        if df is None:
            st.sidebar.error("Failed to read data.")
            return None
        if df.empty:
            st.sidebar.warning("The uploaded file is empty or could not be parsed into data.")
            return df

        # Debugging information
        st.sidebar.info(f"Original DataFrame shape: {df.shape}")
        st.sidebar.info(f"Original columns: {', '.join(original_columns)}")

        # Clean and standardize column names
        df.columns = df.columns.str.strip().str.replace(' ', '_', regex=False).str.replace('[^A-Za-z0-9_]+', '',
                                                                                           regex=True)
        df.columns = [f"col_{i}" if c == "" else c for i, c in enumerate(df.columns)]  # Handle empty names
        standardized_columns = list(df.columns)

        # Store the mapping between standardized and original column names
        if 'column_mapping' not in st.session_state:
            st.session_state['column_mapping'] = {}

        # Update the mapping with new columns
        for std, orig in zip(standardized_columns, original_columns):
            st.session_state['column_mapping'][std] = orig

        # Define expected columns and their types/handling
        date_cols = ['Last_Transaction_Date']
        string_cols_require_str = ["Account_ID", "Account_Type", "Account_Status", "Email_Contact_Attempt",
                                   "SMS_Contact_Attempt", "Phone_Call_Attempt", "KYC_Status", "Branch"]

        # Ensure expected columns exist, add if missing with default value 'Unknown' or NaT for date
        for col in date_cols:
            if col not in df.columns:
                df[col] = pd.NaT
                st.sidebar.warning(f"Missing expected column '{col}'. Added with missing values.")
        for col in string_cols_require_str:
            if col not in df.columns:
                df[col] = 'Unknown'
                st.sidebar.warning(f"Missing expected column '{col}'. Added with 'Unknown' values.")

        # Type conversion and cleaning for expected columns
        for col in date_cols:
            if col in df.columns:  # Check again after potentially adding
                # Show the unique values before conversion for debugging
                if not df[col].empty:
                    unique_vals = df[col].dropna().unique()
                    if len(unique_vals) > 0:
                        st.sidebar.info(f"Sample dates before conversion ({col}): {unique_vals[:3]}")

                # Attempt robust date conversion with multiple formats
                try:
                    # First try standard conversion with error coercing
                    df[col] = pd.to_datetime(df[col], errors='coerce')

                    # Check if we got too many NaT values (>50%)
                    if df[col].isna().mean() > 0.5:
                        st.sidebar.warning(
                            f"Over 50% of dates in '{col}' couldn't be parsed. Trying alternative formats...")

                        # Save a copy of the original column
                        orig_dates = df[col].copy()

                        # Try common date formats explicitly
                        formats = ['%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%Y/%m/%d',
                                   '%d-%m-%Y', '%m-%d-%Y', '%Y.%m.%d', '%d.%m.%Y']

                        for fmt in formats:
                            try:
                                df[col] = pd.to_datetime(orig_dates, format=fmt, errors='coerce')
                                # If this format worked well (less than 25% NaT), use it
                                if df[col].isna().mean() < 0.25:
                                    st.sidebar.info(f"Successfully parsed dates using format: {fmt}")
                                    break
                            except:
                                continue
                except Exception as e:
                    st.sidebar.error(f"Error converting dates in column '{col}': {e}")
                    # Ensure column exists even if conversion failed
                    df[col] = pd.NaT

        for col in string_cols_require_str:
            if col in df.columns:  # Check again after potentially adding
                # Ensure string type and fill NaNs, strip whitespace
                try:
                    df[col] = df[col].astype(str).fillna('Unknown').str.strip()
                    # Replace common 'no data' indicators with 'Unknown'
                    df[col] = df[col].replace(['nan', 'None', '', 'Null', 'NULL', 'null'], 'Unknown', regex=True)
                except Exception as e:
                    st.sidebar.error(f"Error standardizing column '{col}': {e}")
                    # Ensure column exists with default value
                    df[col] = 'Unknown'

        # Final validation check
        if df is None or df.empty:
            st.sidebar.error("Data processing resulted in empty DataFrame. Check input data.")
            return None

        st.sidebar.success(f"✅ Data parsed and standardized successfully! Shape: {df.shape}")
        return df
    except Exception as e:
        st.sidebar.error(f"Error during data parsing/standardization: {e}")
        st.sidebar.error(f"Original columns detected: {original_columns if original_columns else 'N/A'}")
        import traceback
        st.sidebar.error(f"Traceback: {traceback.format_exc()}")
        return None


def save_to_db(df, table_name="accounts_data"):
    """Saves DataFrame to Azure SQL (default connection), replacing table data."""
    if df is None or df.empty:
        st.sidebar.warning(f"Skipped saving empty or None DataFrame to '{table_name}'.")
        return False

    conn = get_db_connection() # Uses the cached connection
    if conn is None:
        st.sidebar.error(f"Cannot save to DB: Default DB connection failed for '{table_name}'.")
        return False

    try:
        required_db_cols = ["Account_ID", "Account_Type", "Last_Transaction_Date", "Account_Status",
                            "Email_Contact_Attempt", "SMS_Contact_Attempt", "Phone_Call_Attempt", "KYC_Status",
                            "Branch"]
        cols_to_save = [col for col in required_db_cols if col in df.columns]
        if not cols_to_save:
            st.sidebar.error(f"No matching columns found in DataFrame for table '{table_name}'. Cannot save.")
            return False

        df_to_save = df[cols_to_save].copy()

        # Convert datetime columns to string format compatible with SQL Server DATETIME2
        for col in df_to_save.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, UTC]']).columns:
             # Convert NaT to None so pyodbc can handle it as SQL NULL
             df_to_save[col] = df_to_save[col].where(pd.notna(df_to_save[col]), None)
             # Format valid dates
             df_to_save[col] = df_to_save[col].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S.%f') if isinstance(x, datetime) else None)

        # Convert other relevant columns to string to handle potential mixed types gracefully during insertion
        for col in ['Account_ID', 'Account_Type', 'Account_Status', 'Email_Contact_Attempt',
                    'SMS_Contact_Attempt', 'Phone_Call_Attempt', 'KYC_Status', 'Branch']:
            if col in df_to_save.columns:
                df_to_save[col] = df_to_save[col].astype(str)
                # Replace 'None' string resulting from NaT or actual None in conversion with 'Unknown' or empty string if appropriate
                # Using 'Unknown' as per parsing logic, but consider if empty string is better for DB NVARCHAR
                df_to_save[col] = df_to_save[col].replace('None', 'Unknown')


        with conn.cursor() as cursor:
            # Check if table exists before truncating
            cursor.execute(f"SELECT 1 FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = ?", (table_name,))
            table_exists = cursor.fetchone() is not None

            if not table_exists:
                st.sidebar.error(f"Table '{table_name}' does not exist in the database. Cannot save.")
                return False

            cursor.execute(f"TRUNCATE TABLE {table_name}")
            # conn.commit() # Commit truncate immediately

            # Prepare for bulk insert or row-by-row insert
            # Row-by-row is simpler but slower for large datasets. For demonstration, it's okay.
            # For production, consider using `executemany` or a dedicated bulk insert library.

            # Build the INSERT statement template
            placeholders = ','.join(['?'] * len(cols_to_save))
            columns_str = ','.join([f'[{c}]' for c in cols_to_save]) # Enclose column names in brackets for safety
            insert_sql = f"INSERT INTO {table_name} ({columns_str}) VALUES ({placeholders})"

            # Prepare values as tuples for executemany
            # Handle potential None values for dates/nullable columns
            values_to_insert = []
            for index, row in df_to_save.iterrows():
                 # Convert pandas NaT to Python None explicitly for pyodbc
                 prepared_values = []
                 for col in cols_to_save:
                     value = row[col]
                     if pd.isna(value): # Check for pandas missing values (NaT for dates, NaN for numeric)
                         prepared_values.append(None)
                     else:
                         prepared_values.append(value) # Keep other types as is (they should be strings/formatted dates)
                 values_to_insert.append(tuple(prepared_values))


            if values_to_insert:
                 # Use executemany for better performance than row-by-row execute
                 cursor.executemany(insert_sql, values_to_insert)

            conn.commit()
        return True
    except pyodbc.Error as e:
        st.sidebar.error(f"Database Save Error ('{table_name}'): {e}. Check data compatibility or constraints.")
        return False
    except KeyError as e:
        st.sidebar.error(f"Missing expected column during save preparation: {e}")
        return False
    except Exception as e:
        st.sidebar.error(f"An unexpected error occurred during DB save ('{table_name}'): {e}")
        return False


def save_summary_to_db(observation, trend, insight, action):
    """Saves analysis summary to the insight log table using default DB connection."""
    conn = get_db_connection() # Uses the cached connection
    if conn is None:
        st.error("Cannot save summary to DB: Default DB connection failed.")
        return False

    try:
        with conn:
            cursor = conn.cursor()
            insert_sql = """
                         INSERT INTO insight_log (timestamp, observation, trend, insight, action)
                         VALUES (?, ?, ?, ?, ?)
                         """
            timestamp = datetime.now()
            # Ensure data types match DB schema NVARCHAR(MAX)
            cursor.execute(insert_sql, (timestamp, str(observation)[:4000], str(trend)[:4000], str(insight)[:4000], str(action)[:4000])) # Truncate to fit MAX size if needed, or ensure column is NVARCHAR(MAX)
            conn.commit()
        return True
    except pyodbc.Error as e:
        st.error(f"Failed to save summary to DB: {e}")
        return False
    except Exception as e:
        st.error(f"An unexpected error occurred saving summary: {e}")
        return False


# === Chatbot Backend Function (Dynamic LLM Approach) ===
# Removed caching from get_response_and_chart as it depends on user_query and current_data,
# and caching the Plotly chart object might not be straightforward or efficient.
def get_response_and_chart(user_query, current_data, llm_model):
    """
    Processes user query dynamically using LLM. Determines if it's a plot request
    or a question, generates the plot or answers accordingly.
    Handles JSON parsing and potential errors.
    """
    chart = None
    response_text = "Sorry, something went wrong processing your request." # Default error message

    if not llm_model or not LANGCHAIN_AVAILABLE:
        return "⚠️ AI Assistant not available (check API key or install Langchain). Cannot process dynamic requests.", None

    if current_data is None or current_data.empty:
        return "⚠️ No data loaded. Please upload and process data first.", None

    # --- Prepare Context for LLM ---
    try:
        cols_info = []
        for col in current_data.columns:
            dtype = str(current_data[col].dtype)
            # Provide more specific info based on dtype
            if pd.api.types.is_numeric_dtype(current_data[col]):
                if current_data[col].notna().any():
                    desc = current_data[col].describe()
                    unique_approx = f"Min: {desc['min']:.2f}, Max: {desc['max']:.2f}, Mean: {desc['mean']:.2f}"
                else:
                    unique_approx = "No numeric data"
            elif pd.api.types.is_datetime64_any_dtype(current_data[col]):
                 if current_data[col].notna().any():
                    unique_approx = f"Date Range: {current_data[col].min().strftime('%Y-%m-%d')} to {current_data[col].max().strftime('%Y-%m-%d')}"
                 else:
                     unique_approx = "No valid dates"
            elif pd.api.types.is_string_dtype(current_data[col]):
                 unique_count = current_data[col].nunique()
                 unique_approx = f"~{unique_count} unique values"
                 if unique_count > 0 and unique_count <= 20: # List categories if few
                     unique_vals = current_data[col].dropna().unique()
                     unique_approx += f" (Values: {', '.join(map(str, unique_vals))})"
                 elif unique_count > 20:
                      unique_approx += f" (Top 3 e.g., {', '.join(map(str, current_data[col].value_counts().nlargest(3).index))})"
            else:
                unique_approx = f"Type: {dtype}" # Generic fallback

            cols_info.append(f"- `{col}` ({unique_approx})")

        columns_description = "\n".join(cols_info)
        num_rows = len(current_data)
        # Allowed plot types relevant to common business data analysis
        allowed_plots = ['bar', 'pie', 'histogram', 'box', 'scatter']

        interpretation_prompt_text = """
You are an intelligent assistant interpreting user requests about a banking compliance dataset.
Analyze the user's query: "{user_query}"
Available Data Context:
- Number of rows: {num_rows}
- Available standardized columns and their details:
{columns_description}
- Allowed plot types (from plotly.express): {allowed_plots_str}
- Pie charts require a categorical column with few (<25) unique values.
- Histograms and Box plots are best for numeric or date columns.
- Bar charts are good for counts by category.
- Scatter plots need two numeric/date columns.

Task:
1. Determine if the user query is primarily a request to **plot** data or a **question** to be answered.
2. If it's a **plotting request** AND seems feasible with the available columns and allowed plot types:
   - Identify the most appropriate plot type from the allowed list ({allowed_plots_str}).
   - Identify the necessary column(s) for that plot type using the **exact standardized column names** provided above.
     - For 'bar': `x_column` (category), optional `color_column`.
     - For 'pie': `names_column` (categorical, few unique values). `values_column` is implicitly the count.
     - For 'histogram': `x_column` (numeric/date), optional `color_column`.
     - For 'box': `y_column` (numeric), optional `x_column` (category), optional `color_column`.
     - For 'scatter': `x_column` (numeric/date), `y_column` (numeric/date), optional `color_column`.
   - Generate a concise, suitable title for the plot based on the query and columns used.
   - Output **ONLY** a valid JSON object with the following structure. Use `null` (JSON null) for unused keys.
     ```json
     {{
       "action": "plot",
       "plot_type": "chosen_plot_type",
       "x_column": "Standardized_X_Column_Name_or_null",
       "y_column": "Standardized_Y_Column_Name_or_null",
       "color_column": "Standardized_Color_Column_Name_or_null",
       "names_column": "Standardized_Pie_Names_Column_or_null",
       "title": "Suggested Plot Title"
     }}
     ```
3. If the query is a **question**, a request for analysis/summary, or an infeasible/unclear plot request:
   - Output **ONLY** a valid JSON object with the structure:
     ```json
     {{
       "action": "answer",
       "query_for_llm": "{user_query}"
     }}
     ```
Constraints:
- Adhere strictly to the JSON format specified for each action.
- Only use standardized column names listed in the context.
- Only use plot types from the allowed list: {allowed_plots_str}.
- If a plot request is ambiguous or infeasible based on column types/values, default to the "answer" action.
- The JSON output must be the *only* text generated. Do NOT add any explanations, greetings, or markdown formatting around the JSON.
JSON Output:
"""
        interpretation_prompt = PromptTemplate.from_template(interpretation_prompt_text)
        prompt_input = {
            "user_query": user_query,
            "num_rows": num_rows,
            "columns_description": columns_description,
            "allowed_plots_str": ', '.join(allowed_plots)
        }
        interpretation_chain = interpretation_prompt | llm_model | StrOutputParser()

        # print(f"DEBUG: Invoking LLM for interpretation...")
        # print(f"DEBUG: Prompt Input: {prompt_input}") # Log prompt input for debugging
        with st.spinner("Interpreting request..."):
            llm_json_output_str = interpretation_chain.invoke(prompt_input)
            # print(f"DEBUG: LLM Interpretation Output (raw): {llm_json_output_str}") # Log raw output

        try:
            # Clean the output - sometimes LLMs wrap JSON in ```json ... ```
            cleaned_json_str = re.sub(r"^```json\s*|\s*```$", "", llm_json_output_str, flags=re.MULTILINE).strip()
            if not cleaned_json_str: raise ValueError("LLM returned an empty response after cleaning.")
            llm_output = json.loads(cleaned_json_str)
            action = llm_output.get("action")
            # print(f"DEBUG: Parsed LLM Action: {action}")

            if action == "plot":
                plot_type = llm_output.get("plot_type")
                x_col = llm_output.get("x_column")
                y_col = llm_output.get("y_column")
                color_col = llm_output.get("color_column")
                names_col = llm_output.get("names_column")
                title = llm_output.get("title", f"Plot based on: {user_query[:40]}...") # Truncate title for brevity
                all_cols = list(current_data.columns)

                # Validate columns exist in the DataFrame
                def validate_col(col_name):
                    return col_name if col_name is not None and col_name in all_cols else None

                x_col_valid = validate_col(x_col)
                y_col_valid = validate_col(y_col)
                color_col_valid = validate_col(color_col)
                names_col_valid = validate_col(names_col)

                # print(f"DEBUG: Plot Params (Validated) - Type:'{plot_type}', X:'{x_col_valid}', Y:'{y_col_valid}', Color:'{color_col_valid}', Names:'{names_col_valid}'")

                # --- Plotting Logic ---
                if plot_type not in allowed_plots:
                     raise ValueError(f"Suggested plot type '{plot_type}' is not in the allowed list: {', '.join(allowed_plots)}")

                if plot_type == 'pie':
                    if not names_col_valid: raise ValueError("Valid 'names_column' needed for pie chart.")
                    # Ensure the column is suitable for pie (categorical with limited unique values)
                    if not pd.api.types.is_string_dtype(current_data[names_col_valid]) and not pd.api.types.is_categorical_dtype(current_data[names_col_valid]):
                         raise ValueError(f"Pie chart 'names_column' ('{names_col_valid}') must be categorical.")
                    unique_count = current_data[names_col_valid].nunique()
                    if unique_count > 25: raise ValueError(
                        f"Too many unique values ({unique_count}) in '{names_col_valid}' for a pie chart (max 25 recommended). Try a bar chart instead.")

                    # Calculate counts for the pie chart
                    counts = current_data[names_col_valid].value_counts().reset_index()
                    counts.columns = [names_col_valid, 'count'] # Rename columns for px.pie
                    chart = px.pie(counts, names=names_col_valid, values='count', title=title, hole=0.3) # Added hole
                    response_text = f"Generated pie chart showing distribution of '{st.session_state['column_mapping'].get(names_col_valid, names_col_valid)}'." # Use original name if available

                elif plot_type == 'bar':
                    if not x_col_valid: raise ValueError("Valid 'x_column' needed for bar chart.")
                    # Bar charts typically show counts per category or sum of y per category
                    # px.histogram is often easier for counts of a categorical variable
                    if pd.api.types.is_numeric_dtype(current_data[x_col_valid]):
                        # If x is numeric, might need a y for sums/means, or binning (histogram better)
                         raise ValueError(f"Bar chart 'x_column' ('{x_col_valid}') is numeric. Consider 'histogram' or provide a categorical 'x_column'.")
                    # Calculate counts per category
                    counts = current_data[x_col_valid].value_counts().reset_index()
                    counts.columns = [x_col_valid, 'count']
                    chart = px.bar(counts, x=x_col_valid, y='count', color=color_col_valid, title=title)
                    response_text = f"Generated bar chart showing counts for '{st.session_state['column_mapping'].get(x_col_valid, x_col_valid)}'" + (
                        f" colored by '{st.session_state['column_mapping'].get(color_col_valid, color_col_valid)}'." if color_col_valid else ".")

                elif plot_type == 'histogram':
                    if not x_col_valid: raise ValueError("Valid 'x_column' needed for histogram.")
                    if not pd.api.types.is_numeric_dtype(current_data[x_col_valid]) and not pd.api.types.is_datetime64_any_dtype(current_data[x_col_valid]):
                        raise ValueError(
                            f"Histogram requires a numeric or date column. '{st.session_state['column_mapping'].get(x_col_valid, x_col_valid)}' is not. Try a 'bar chart' instead for counts.")
                    chart = px.histogram(current_data, x=x_col_valid, color=color_col_valid, title=title)
                    response_text = f"Generated histogram for '{st.session_state['column_mapping'].get(x_col_valid, x_col_valid)}'" + (
                        f" colored by '{st.session_state['column_mapping'].get(color_col_valid, color_col_valid)}'." if color_col_valid else ".")

                elif plot_type == 'box':
                    if not y_col_valid: raise ValueError("Valid 'y_column' (numeric) needed for box plot.")
                    if not pd.api.types.is_numeric_dtype(current_data[y_col_valid]): raise ValueError(
                        f"Box plot requires a numeric 'y_column'. '{st.session_state['column_mapping'].get(y_col_valid, y_col_valid)}' is not numeric.")
                    chart = px.box(current_data, x=x_col_valid, y=y_col_valid, color=color_col_valid, title=title,
                                   points="outliers")
                    response_text = f"Generated box plot for '{st.session_state['column_mapping'].get(y_col_valid, y_col_valid)}'" + (
                        f" grouped by '{st.session_state['column_mapping'].get(x_col_valid, x_col_valid)}'." if x_col_valid else "") + (
                                        f" colored by '{st.session_state['column_mapping'].get(color_col_valid, color_col_valid)}'." if color_col_valid else ".")

                elif plot_type == 'scatter':
                    if not x_col_valid or not y_col_valid: raise ValueError(
                        f"Valid numeric/date 'x_column' ('{x_col_valid}') and 'y_column' ('{y_col_valid}') needed for scatter plot.")
                    if not (pd.api.types.is_numeric_dtype(current_data[x_col_valid]) or pd.api.types.is_datetime64_any_dtype(current_data[x_col_valid])):
                         raise ValueError(f"Scatter plot 'x_column' ('{st.session_state['column_mapping'].get(x_col_valid, x_col_valid)}') must be numeric or date.")
                    if not (pd.api.types.is_numeric_dtype(current_data[y_col_valid]) or pd.api.types.is_datetime64_any_dtype(current_data[y_col_valid])):
                         raise ValueError(f"Scatter plot 'y_column' ('{st.session_state['column_mapping'].get(y_col_valid, y_col_valid)}') must be numeric or date.")

                    chart = px.scatter(current_data, x=x_col_valid, y=y_col_valid, color=color_col_valid, title=title,
                                       hover_data=current_data.columns)
                    response_text = f"Generated scatter plot of '{st.session_state['column_mapping'].get(x_col_valid, x_col_valid)}' vs '{st.session_state['column_mapping'].get(y_col_valid, y_col_valid)}'" + (
                        f" colored by '{st.session_state['column_mapping'].get(color_col_valid, color_col_valid)}'." if color_col_valid else ".")

                # --- Add Summary Stats if Plot Successful ---
                # Identify the primary column used in the plot for summary
                primary_plot_col = names_col_valid or x_col_valid or y_col_valid
                if chart and primary_plot_col and primary_plot_col in current_data.columns:
                    temp_data_for_stats = current_data[primary_plot_col].dropna()
                    summary_text = ""
                    if pd.api.types.is_numeric_dtype(temp_data_for_stats) and not temp_data_for_stats.empty:
                        desc = temp_data_for_stats.describe()
                        summary_text = (f"\n\n**Summary for '{st.session_state['column_mapping'].get(primary_plot_col, primary_plot_col)}' ({primary_plot_col}):** "
                                        f"Mean: {desc.get('mean', float('nan')):.2f}, "
                                        f"Std: {desc.get('std', float('nan')):.2f}, "
                                        f"Min: {desc.get('min', float('nan')):.2f}, "
                                        f"Max: {desc.get('max', float('nan')):.2f}, "
                                        f"Count: {int(desc.get('count', 0))}")
                    elif pd.api.types.is_datetime64_any_dtype(temp_data_for_stats) and not temp_data_for_stats.empty:
                         summary_text = (f"\n\n**Summary for '{st.session_state['column_mapping'].get(primary_plot_col, primary_plot_col)}' ({primary_plot_col}):** "
                                         f"Earliest: {temp_data_for_stats.min().strftime('%Y-%m-%d')}, "
                                         f"Latest: {temp_data_for_stats.max().strftime('%Y-%m-%d')}, "
                                         f"Count: {len(temp_data_for_stats)}")
                    elif not temp_data_for_stats.empty:
                        counts = temp_data_for_stats.value_counts()
                        top_categories = [f"'{str(i)}' ({counts[i]})" for i in counts.head(3).index]
                        summary_text = (f"\n\n**Summary for '{st.session_state['column_mapping'].get(primary_plot_col, primary_plot_col)}' ({primary_plot_col}):** {counts.size} unique values. "
                                        f"Top: {', '.join(top_categories)}.")
                    response_text += summary_text # Append summary to the response text

            elif action == "answer":
                query_to_answer = llm_output.get("query_for_llm", user_query)
                # print(f"DEBUG: Invoking LLM for answering: {query_to_answer}")
                col_context = f"Dataset has {len(current_data)} rows. Standardized columns available: {', '.join(current_data.columns)}. "
                if 'column_mapping' in st.session_state and st.session_state['column_mapping']:
                    # Show original names in context for better understanding by LLM
                    original_names_map = {st.session_state['column_mapping'][std_col]: std_col for std_col in current_data.columns if std_col in st.session_state['column_mapping']}
                    if original_names_map:
                         col_context += f"Original column names and their standardized versions: {'; '.join([f'{orig} (std: {std})' for orig, std in original_names_map.items()])}. "

                answer_prompt_text = """
You are a helpful banking compliance assistant. Answer the user's question based on the provided context about the loaded dataset.
Be concise and directly address the question. If the question asks for specific numbers or analysis not directly available from simple column value counts, state that you can only provide insights based on the available columns and cannot perform complex calculations on the entire dataset interactively.

Context about the dataset:
{data_context}

User Question: {user_question}

Answer:"""
                answer_prompt = PromptTemplate.from_template(answer_prompt_text)
                answer_chain = answer_prompt | llm_model | StrOutputParser()
                with st.spinner("🤔 Thinking..."):
                    ai_response_content = answer_chain.invoke(
                        {"data_context": col_context, "user_question": query_to_answer})
                response_text = ai_response_content if ai_response_content else "Sorry, I couldn't formulate an answer."
                chart = None # Ensure chart is None for answers

            else:
                response_text = f"Sorry, I received an unexpected instruction ('{action}') from the AI interpreter. Please rephrase your request."
                chart = None # Ensure chart is None

        except (json.JSONDecodeError, ValueError) as e:
            print(f"ERROR: LLM output processing failed: {e}. Raw output: '{llm_json_output_str}'")
            response_text = f"Sorry, there was an error processing the AI's response ({e}). It might not have returned the expected format. Please try rephrasing."
            chart = None # Ensure chart is None
        except Exception as e:
            print(f"ERROR: Unexpected error during AI processing or plotting: {e}")
            response_text = f"❌ An unexpected error occurred: {e}"
            chart = None # Ensure chart is None

    except Exception as e:
        # Catch errors during prompt preparation or initial LLM invocation
        print(f"ERROR: Failed to invoke LLM or prepare request context: {e}")
        response_text = f"❌ Failed to process your request due to an internal error: {e}"
        chart = None # Ensure chart is None

    return response_text, chart


# === Helper Functions for Agent Detection ===
# These functions were already robust in taking df and threshold as input.
# No changes needed here, they operate on the st.session_state.app_df copy.

def check_safe_deposit(df, threshold_date):
    """Detects safe deposit accounts inactive over threshold with no contact attempts."""
    try:
        # Ensure columns exist and handle potential None/NaN in string comparisons
        if not all(col in df.columns for col in ['Account_Type', 'Last_Transaction_Date', 'Email_Contact_Attempt', 'SMS_Contact_Attempt', 'Phone_Call_Attempt']):
             return pd.DataFrame(), 0, "(Skipped: Required columns missing for Safe Deposit check)"

        data = df[
            (df['Account_Type'].astype(str).str.contains("Safe Deposit", case=False, na=False)) &
            (df['Last_Transaction_Date'].notna()) & # Ensure date is not NaT
            (df['Last_Transaction_Date'] < threshold_date) &
            (df['Email_Contact_Attempt'].astype(str).str.lower() == 'no') & # Handle NaNs by converting to str
            (df['SMS_Contact_Attempt'].astype(str).str.lower() == 'no') &
            (df['Phone_Call_Attempt'].astype(str).str.lower() == 'no')
        ]
        count = len(data)
        desc = f"Safe Deposit without contact attempts (>3y): {count} accounts"
        return data, count, desc
    except Exception as e:
        return pd.DataFrame(), 0, f"(Error in Safe Deposit check: {e})"


def check_investment_inactivity(df, threshold_date):
    """Detects investment accounts inactive over threshold with no contact attempts."""
    try:
        if not all(col in df.columns for col in ['Account_Type', 'Last_Transaction_Date', 'Email_Contact_Attempt', 'SMS_Contact_Attempt', 'Phone_Call_Attempt']):
             return pd.DataFrame(), 0, "(Skipped: Required columns missing for Investment check)"

        data = df[
            (df['Account_Type'].astype(str).str.contains("Investment", case=False, na=False)) &
            (df['Last_Transaction_Date'].notna()) &
            (df['Last_Transaction_Date'] < threshold_date) &
            (df['Email_Contact_Attempt'].astype(str).str.lower() == 'no') &
            (df['SMS_Contact_Attempt'].astype(str).str.lower() == 'no') &
            (df['Phone_Call_Attempt'].astype(str).str.lower() == 'no')
        ]
        count = len(data)
        desc = f"Investment accounts without activity or contact (>3y): {count} accounts"
        return data, count, desc
    except Exception as e:
        return pd.DataFrame(), 0, f"(Error in Investment inactivity check: {e})"


def check_fixed_deposit_inactivity(df, threshold_date):
    """Detects fixed deposit accounts inactive over threshold."""
    try:
        if not all(col in df.columns for col in ['Account_Type', 'Last_Transaction_Date']):
             return pd.DataFrame(), 0, "(Skipped: Required columns missing for Fixed Deposit check)"

        data = df[
            (df['Account_Type'].astype(str).str.lower() == 'fixed deposit') &
            (df['Last_Transaction_Date'].notna()) &
            (df['Last_Transaction_Date'] < threshold_date)
        ]
        count = len(data)
        desc = f"Fixed deposit accounts with no activity (>3y): {count} accounts"
        return data, count, desc
    except Exception as e:
        return pd.DataFrame(), 0, f"(Error in Fixed Deposit inactivity check: {e})"


def check_general_inactivity(df, threshold_date):
    """Detects Savings/Call/Current accounts inactive over threshold."""
    try:
        if not all(col in df.columns for col in ['Account_Type', 'Last_Transaction_Date']):
             return pd.DataFrame(), 0, "(Skipped: Required columns missing for General Inactivity check)"

        data = df[
            (df['Account_Type'].astype(str).isin(["Savings", "Call", "Current"])) &
            (df['Last_Transaction_Date'].notna()) &
            (df['Last_Transaction_Date'] < threshold_date)
        ]
        count = len(data)
        desc = f"General accounts (Savings/Call/Current) with no activity (>3y): {count} accounts"
        return data, count, desc
    except Exception as e:
        return pd.DataFrame(), 0, f"(Error in General inactivity check: {e})"


def check_unreachable_dormant(df):
    """Detects accounts marked dormant with no contact attempts."""
    try:
        if not all(col in df.columns for col in ['Account_Status', 'Email_Contact_Attempt', 'SMS_Contact_Attempt', 'Phone_Call_Attempt']):
             return pd.DataFrame(), 0, "(Skipped: Required columns missing for Unreachable Dormant check)"

        data = df[
            (df['Account_Status'].astype(str).str.lower() == 'dormant') &
            (df['Email_Contact_Attempt'].astype(str).str.lower() == 'no') &
            (df['SMS_Contact_Attempt'].astype(str).str.lower() == 'no') &
            (df['Phone_Call_Attempt'].astype(str).str.lower() == 'no')
        ]
        count = len(data)
        desc = f"Unreachable accounts already marked dormant: {count} accounts"
        return data, count, desc
    except Exception as e:
        return pd.DataFrame(), 0, f"(Error in Unreachable dormant check: {e})"


# === Compliance Agents ===
def detect_incomplete_contact(df):
    """Detects accounts with at least one 'No' contact attempt."""
    try:
        if not all(col in df.columns for col in ['Email_Contact_Attempt', 'SMS_Contact_Attempt', 'Phone_Call_Attempt']):
             return pd.DataFrame(), 0, "(Skipped: Required columns missing for Incomplete Contact check)"

        data = df[
            (df['Email_Contact_Attempt'].astype(str).str.lower() == 'no') |
            (df['SMS_Contact_Attempt'].astype(str).str.lower() == 'no') |
            (df['Phone_Call_Attempt'].astype(str).str.lower() == 'no')
        ]
        count = len(data)
        desc = f"Accounts with incomplete contact attempts: {count} accounts"
        return data, count, desc
    except Exception as e:
        return pd.DataFrame(), 0, f"(Error in contact attempt verification: {e})"


def detect_flag_candidates(df, threshold_date):
    """Detects accounts inactive over threshold, not yet flagged dormant."""
    try:
        if not all(col in df.columns for col in ['Last_Transaction_Date', 'Account_Status']):
             return pd.DataFrame(), 0, "(Skipped: Required columns missing for Flag Candidate check)"

        data = df[
            (df['Last_Transaction_Date'].notna()) &
            (df['Last_Transaction_Date'] < threshold_date) &
            (df['Account_Status'].astype(str).str.lower() != 'dormant')
        ]
        count = len(data)
        desc = f"Accounts inactive over threshold, not yet flagged dormant: {count} accounts"
        return data, count, desc
    except Exception as e:
        return pd.DataFrame(), 0, f"(Error in dormant flag detection: {e})"


def detect_ledger_candidates(df):
    """Detects accounts marked dormant requiring ledger classification (not already in ledger)."""
    try:
        if 'Account_Status' not in df.columns:
             return pd.DataFrame(), 0, "(Skipped: Required column missing for Ledger Candidate check)"

        data = df[df['Account_Status'].astype(str).str.lower() == 'dormant'].copy() # Work on copy to avoid SettingWithCopyWarning

        ids_in_ledger = []
        conn = get_db_connection() # Uses cached connection
        if conn: # Only attempt DB check if connection is available
            try:
                with conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT account_id FROM dormant_ledger")
                    ids_in_ledger = [row[0] for row in cursor.fetchall()]
            except Exception as db_e:
                st.warning(f"Could not check dormant ledger table: {db_e}. Proceeding without filtering.")
                ids_in_ledger = [] # Ensure it's an empty list if check fails

        if ids_in_ledger:
            data = data[~data['Account_ID'].isin(ids_in_ledger)]

        count = len(data)
        desc = f"Dormant accounts requiring ledger classification (not yet in ledger): {count} accounts"
        return data, count, desc
    except Exception as e:
        return pd.DataFrame(), 0, f"(Error in ledger candidate detection: {e})"


def detect_freeze_candidates(df, threshold_date):
    """Detects dormant accounts inactive beyond freeze threshold."""
    try:
        if not all(col in df.columns for col in ['Last_Transaction_Date', 'Account_Status']):
             return pd.DataFrame(), 0, "(Skipped: Required columns missing for Freeze Candidate check)"

        data = df[
            (df['Last_Transaction_Date'].notna()) &
            (df['Last_Transaction_Date'] < threshold_date) &
            (df['Account_Status'].astype(str).str.lower() == 'dormant')
        ]
        count = len(data)
        desc = f"Dormant accounts inactive beyond freeze threshold: {count} accounts"
        return data, count, desc
    except Exception as e:
        return pd.DataFrame(), 0, f"(Error in freeze candidate detection: {e})"


def detect_transfer_candidates(df, cutoff_date):
    """Detects dormant accounts inactive before a specific cutoff date (e.g., CBUAE)."""
    if not isinstance(cutoff_date, datetime):
        return pd.DataFrame(), 0, "(Skipped: Valid cutoff date not provided for Transfer check)"
    try:
        if not all(col in df.columns for col in ['Last_Transaction_Date', 'Account_Status']):
             return pd.DataFrame(), 0, "(Skipped: Required columns missing for Transfer Candidate check)"

        data = df[
            (df['Last_Transaction_Date'].notna()) &
            (df['Last_Transaction_Date'] < cutoff_date) &
            (df['Account_Status'].astype(str).str.lower() == 'dormant')
        ]
        count = len(data)
        desc = f"Dormant accounts inactive before cutoff ({cutoff_date.strftime('%Y-%m-%d')}): {count} accounts"
        return data, count, desc
    except Exception as e:
        return pd.DataFrame(), 0, f"(Error in transfer candidate detection: {e})"


# === Main Application ===
def main():
    st.sidebar.header("📤 Data Upload")
    # --- Updated Radio Button Labels for Clarity ---
    upload_method = st.sidebar.radio("Select upload method:",
                                     ["**Upload File (CSV/XLSX/JSON)**", "**Upload via URL**",
                                      "**Load Data from Azure SQL (into App)**"], # Clarified purpose
                                     key="upload_method_radio")

    # Session state initialization
    if 'app_df' not in st.session_state: st.session_state.app_df = None
    if 'data_processed' not in st.session_state: st.session_state.data_processed = False
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [{"role": "assistant", "content": "Hi! Please upload data first..."}]
    if 'column_mapping' not in st.session_state: st.session_state.column_mapping = {}
    # Removed connected_via_sql_manual state as SQL Bot now uses default connection

    uploaded_data_source = None
    if upload_method == "**Upload File (CSV/XLSX/JSON)**":
        uploaded_file = st.sidebar.file_uploader("Upload Account Dataset", type=["csv", "xlsx", "xls", "json"],
                                                 key="data_file_uploader")
        if uploaded_file:
            uploaded_data_source = uploaded_file
            st.sidebar.caption(f"Selected: {uploaded_file.name}")
    elif upload_method == "**Upload via URL**":
        url_input = st.sidebar.text_input("Enter CSV file URL:", key="url_input")
        if st.sidebar.button("Fetch Data from URL", key="fetch_url_button"):
            if url_input:
                try:
                    with st.spinner("⏳ Fetching data from URL..."):
                        response = requests.get(url_input, timeout=30)
                        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
                        # Assume URL points to a raw CSV file content
                        uploaded_data_source = response.text
                        st.sidebar.success("✅ Fetched! Ready to process.")
                except requests.exceptions.RequestException as e:
                    st.sidebar.error(f"❌ URL Fetch Error: {e}")
                except Exception as e:
                    st.sidebar.error(f"❌ Error processing URL data: {e}")
            else:
                st.sidebar.warning("⚠️ Please enter a valid URL")
    # --- Clarified Azure SQL Load Option ---
    # Fix for the "Load Data from Azure SQL" feature in main() function
    # This is a replacement for the relevant section in the main() function

    # Updated "Load Data from Azure SQL" section with fixes to keep connection open and properly process data

    # Updated "Load Data from Azure SQL" section with fixed variable scope issue

    elif upload_method == "**Load Data from Azure SQL (into App)**":

        st.sidebar.subheader("Azure SQL Data Loader")

        st.sidebar.markdown("*(Connect to a specific DB/table to load data into the application's memory.)*")

        use_secrets_for_conn_upload = st.sidebar.checkbox("Use credentials from secrets.toml (for loading)", value=True,

                                                          key="use_secrets_checkbox_upload")

        # Default values for manual input fields, initialized even if using secrets

        input_db_server_upload, input_db_name_upload = DB_SERVER, DB_NAME

        input_db_username_upload, input_db_password_upload = "", ""

        input_use_entra_upload, input_entra_domain_upload = False, ""

        # Display Manual Input Fields if use_secrets is False

        if not use_secrets_for_conn_upload:

            st.sidebar.warning("⚠️ Secrets credentials disabled. Please enter manual connection details.")

            input_db_server_upload = st.sidebar.text_input("Azure SQL Server (Loader):", value=DB_SERVER,
                                                           key="db_server_input_upload")

            input_db_name_upload = st.sidebar.text_input("Database Name (Loader):", value=DB_NAME,
                                                         key="db_name_input_upload")

            input_db_username_upload = st.sidebar.text_input("Username (Loader):", key="db_username_input_upload")

            input_db_password_upload = st.sidebar.text_input("Password (Loader):", type="password",
                                                             key="db_password_input_upload")

            input_use_entra_upload = st.sidebar.checkbox("Use Microsoft Entra Authentication (Loader)",
                                                         key="use_entra_checkbox_upload")

            if input_use_entra_upload:
                input_entra_domain_upload = st.sidebar.text_input("Microsoft Entra Domain (Loader):",
                                                                  key="entra_domain_input_upload",

                                                                  placeholder="e.g., yourdomain.onmicrosoft.com")

        else:

            # If using secrets, show a message indicating this

            st.sidebar.info("Using credentials from secrets.toml/env vars.")

            st.sidebar.info(f"Default Server: {DB_SERVER}")

            st.sidebar.info(f"Default Database: {DB_NAME}")

        sql_query_input_upload = st.sidebar.text_area("SQL Query to Load Data:", value="SELECT * FROM accounts_data",
                                                      height=100,

                                                      key="azure_sql_query_input_upload")

        # Advanced troubleshooting options

        with st.sidebar.expander("Advanced Connection Options"):

            use_port_in_conn = st.checkbox("Include port in connection string", value=True,
                                           key="use_port_in_conn_upload")

            driver_version = st.selectbox("ODBC Driver Version",
                                          ["ODBC Driver 18 for SQL Server", "ODBC Driver 17 for SQL Server",
                                           "SQL Server Native Client 11.0"], key="driver_version_upload")

            trust_server_cert = st.checkbox("Trust Server Certificate", value=False, key="trust_server_cert_upload")

            timeout_seconds = st.number_input("Connection Timeout (seconds)", min_value=15, max_value=300, value=60,
                                              key="timeout_seconds_upload")

        if st.sidebar.button("Connect & Load Data", key="connect_azure_sql_button_upload"):

            if sql_query_input_upload:

                with st.spinner("⏳ Connecting to Azure SQL Database and loading data..."):

                    conn_upload = None

                    try:

                        conn_str_upload = ""

                        if use_secrets_for_conn_upload:

                            # GET CREDENTIALS DIRECTLY FROM SECRETS/ENV FOR THIS TEMPORARY CONNECTION

                            db_username_secrets = None

                            db_password_secrets = None

                            # Try secrets first

                            if hasattr(st, 'secrets'):
                                db_username_secrets = st.secrets.get("DB_USERNAME")

                                db_password_secrets = st.secrets.get("DB_PASSWORD")

                                use_entra_secrets_str = st.secrets.get("USE_ENTRA_AUTH", "false")

                                use_entra_secrets = use_entra_secrets_str.lower() == "true"

                                entra_domain_secrets = st.secrets.get("ENTRA_DOMAIN")

                            # Fallback to environment variables

                            if not db_username_secrets or not db_password_secrets:

                                db_username_secrets = os.getenv("DB_USERNAME")

                                db_password_secrets = os.getenv("DB_PASSWORD")

                                if db_username_secrets is None or db_password_secrets is None:
                                    st.sidebar.error("DB credentials missing in secrets/env vars for loading.")

                                    raise ValueError("Credentials missing")

                                use_entra_secrets_str = os.getenv("USE_ENTRA_AUTH", "false")

                                use_entra_secrets = use_entra_secrets_str.lower() == "true"

                                entra_domain_secrets = os.getenv("ENTRA_DOMAIN")

                            # Use the DB_SERVER and DB_NAME constants for the secrets connection attempt

                            if use_entra_secrets:

                                if not entra_domain_secrets:
                                    st.sidebar.error(
                                        "Entra auth specified, but ENTRA_DOMAIN is missing in secrets/env.")

                                    raise ValueError("ENTRA_DOMAIN missing")

                                conn_str_upload = (

                                    f"DRIVER={{{driver_version}}};"

                                    f"SERVER={DB_SERVER};"  # Use constants

                                    f"DATABASE={DB_NAME};"

                                    f"Authentication=ActiveDirectoryPassword;"

                                    f"UID={db_username_secrets}@{entra_domain_secrets};PWD={db_password_secrets};"

                                    f"Encrypt=yes;TrustServerCertificate={'yes' if trust_server_cert else 'no'};Connection Timeout={timeout_seconds};"

                                )

                            else:

                                server_with_port = f"{DB_SERVER},{DB_PORT}" if use_port_in_conn else DB_SERVER

                                conn_str_upload = (

                                    f"DRIVER={{{driver_version}}};"

                                    f"SERVER={server_with_port};DATABASE={DB_NAME};"  # Use constants

                                    f"UID={db_username_secrets};PWD={db_password_secrets};"

                                    f"Encrypt=yes;TrustServerCertificate={'yes' if trust_server_cert else 'no'};Connection Timeout={timeout_seconds};"

                                )


                        else:  # Use manual inputs from the sidebar fields

                            if input_use_entra_upload:

                                if not input_entra_domain_upload:
                                    st.sidebar.error(
                                        "Entra auth selected manually, but Microsoft Entra Domain is missing.")

                                    raise ValueError("Manual ENTRA_DOMAIN missing")

                                conn_str_upload = (

                                    f"DRIVER={{{driver_version}}};"

                                    f"SERVER={input_db_server_upload};"  # Use manual inputs

                                    f"DATABASE={input_db_name_upload};"

                                    f"Authentication=ActiveDirectoryPassword;"

                                    f"UID={input_db_username_upload}@{input_entra_domain_upload};PWD={input_db_password_upload};"

                                    f"Encrypt=yes;TrustServerCertificate={'yes' if trust_server_cert else 'no'};Connection Timeout={timeout_seconds};"

                                )

                            else:

                                server_with_port = f"{input_db_server_upload},{DB_PORT}" if use_port_in_conn else input_db_server_upload

                                conn_str_upload = (

                                    f"DRIVER={{{driver_version}}};"

                                    f"SERVER={server_with_port};DATABASE={input_db_name_upload};"  # Use manual inputs

                                    f"UID={input_db_username_upload};PWD={input_db_password_upload};"

                                    f"Encrypt=yes;TrustServerCertificate={'yes' if trust_server_cert else 'no'};Connection Timeout={timeout_seconds};"

                                )

                        # Log the connection string with password obfuscated for debugging

                        debug_conn_str = conn_str_upload

                        if use_secrets_for_conn_upload and db_password_secrets:

                            debug_conn_str = conn_str_upload.replace(db_password_secrets, "*****")

                        elif not use_secrets_for_conn_upload and input_db_password_upload:

                            debug_conn_str = conn_str_upload.replace(input_db_password_upload, "*****")

                        st.sidebar.info(f"Connecting with: {debug_conn_str}")

                        # Attempt the connection using the constructed string

                        conn_upload = pyodbc.connect(conn_str_upload)

                        # Connection Successful, Proceed to Load Data

                        if conn_upload:

                            try:

                                # First check if the query can be executed without errors

                                test_cursor = conn_upload.cursor()

                                test_cursor.execute(sql_query_input_upload)

                                test_cursor.close()

                                # Now use pd.read_sql_query to get the data as a DataFrame

                                df_from_sql = pd.read_sql_query(sql_query_input_upload, conn_upload)

                                if df_from_sql.empty:

                                    st.sidebar.warning("Query returned no results.")

                                    uploaded_data_source = None

                                else:

                                    # Save the DataFrame to uploaded_data_source

                                    uploaded_data_source = df_from_sql.copy()

                                    st.sidebar.success(f"✅ Query successful! Retrieved {len(df_from_sql)} rows.")

                                    # Display a preview of the data in the sidebar

                                    with st.sidebar.expander("Preview Data"):

                                        st.dataframe(df_from_sql.head(5))

                                    # Add an automatic process button

                                    if st.sidebar.button("Process this data now", key="auto_process_sql_data"):

                                        with st.spinner("⏳ Processing and standardizing data..."):

                                            df_parsed = parse_data(uploaded_data_source)

                                        if df_parsed is not None and not df_parsed.empty:

                                            st.session_state.app_df = df_parsed

                                            st.session_state.data_processed = True

                                            st.sidebar.success("✅ Data processed automatically!")

                                            st.rerun()  # Rerun to refresh the UI

                                        else:

                                            st.sidebar.error(
                                                "Could not process the loaded data. Use the 'Process Uploaded/Fetched Data' button to try again.")

                                    else:

                                        st.sidebar.info(
                                            "Click 'Process Uploaded/Fetched Data' button at the top of the sidebar to use this data.")


                            except pyodbc.Error as sql_e:

                                st.sidebar.error(f"❌ SQL Query Error: {sql_e}")

                                uploaded_data_source = None

                        else:

                            st.sidebar.error("❌ Failed to establish database connection for loading.")

                            uploaded_data_source = None


                    except pyodbc.Error as e:

                        st.sidebar.error(f"❌ DB Connection Error: {e}")

                        # Provide more specific error guidance

                        if "08001" in str(e):

                            st.sidebar.warning(
                                "Cannot reach the server. Check server name, firewall rules, and network connection.")

                        elif "28000" in str(e):

                            st.sidebar.warning("Login failed. Check username and password.")

                        elif "42000" in str(e):

                            st.sidebar.warning(
                                "Database access error. Check if the database exists and user has permission.")

                        elif "01000" in str(e) and "TLS" in str(e):

                            st.sidebar.warning(
                                "SSL/TLS error. Try enabling 'Trust Server Certificate' in Advanced Connection Options.")

                        uploaded_data_source = None

                    except ValueError as e:

                        st.sidebar.error(f"❌ Configuration Error: {e}")

                        uploaded_data_source = None

                    except Exception as e:

                        st.sidebar.error(f"❌ An unexpected error occurred during data loading: {e}")

                        uploaded_data_source = None

                    finally:

                        # Only close connection after we're completely done with it

                        if conn_upload:

                            try:

                                conn_upload.close()

                                st.sidebar.success("Connection closed successfully.")

                            except Exception as e:

                                st.sidebar.warning(f"Failed to close loading connection: {e}")


            else:

                st.sidebar.warning("⚠️ Please enter an SQL query.")


    # --- Data Processing Button ---
    # Enable process button only if a data source is available (file, url string, or dataframe from sql load)
    # Updated process button logic to ensure proper data flow and processing

    # --- Data Processing Button ---
    # Enable process button only if a data source is available
    process_button_disabled = uploaded_data_source is None
    process_clicked = st.sidebar.button("Process Uploaded/Fetched Data", key="process_data_button",
                                        disabled=process_button_disabled)

    if process_clicked and uploaded_data_source is not None:
        with st.spinner("⏳ Processing and standardizing data..."):
            # Show a progress message
            progress_text = st.sidebar.empty()
            progress_text.info("Starting data processing...")

            # Try to make a copy of the data to avoid reference issues
            try:
                if isinstance(uploaded_data_source, pd.DataFrame):
                    data_copy = uploaded_data_source.copy()
                    progress_text.info("Made copy of uploaded DataFrame...")
                else:
                    data_copy = uploaded_data_source
                    progress_text.info("Using uploaded data source directly...")

                # Process the data
                progress_text.info("Parsing data...")
                df_parsed = parse_data(data_copy)  # This function is cached
            except Exception as e:
                st.sidebar.error(f"Error preparing data for processing: {e}")
                import traceback
                st.sidebar.error(f"Traceback: {traceback.format_exc()}")
                df_parsed = None

        if df_parsed is not None and not df_parsed.empty:
            # Data was successfully parsed
            st.session_state.app_df = df_parsed
            st.session_state.data_processed = True

            # Prepare initial chat message
            cols_info = []
            for col in df_parsed.columns:
                orig_name = st.session_state['column_mapping'].get(col, col)
                cols_info.append(f"`{orig_name}`")

            std_cols_example = ', '.join(cols_info[:min(5, len(cols_info))])
            initial_message = (f"Data ({len(df_parsed)} rows) processed! Look for columns like: {std_cols_example}...\n"
                               f"You can now use the other modes for analysis.")
            st.session_state.chat_messages = [{"role": "assistant", "content": initial_message}]

            # Attempt to save processed data to the default database
            if db_initialized:  # Only try to save if DB init was successful
                with st.spinner("💾 Saving processed data to default Azure SQL DB..."):
                    save_success = save_to_db(st.session_state.app_df)
                if save_success:
                    st.sidebar.success(f"✅ Processed data saved to default DB!")
                else:
                    st.sidebar.error("Processed data, but failed to save to default DB.")
            else:
                st.sidebar.warning("Skipped saving data to DB because default DB connection/initialization failed.")

            # Show success message
            st.sidebar.success("✅ Data processed successfully!")
            # Use st.balloons for a little celebration
            st.balloons()

        elif df_parsed is not None and df_parsed.empty:
            st.sidebar.error("Source data resulted in an empty dataset after parsing.")
            st.session_state.data_processed = False
            st.session_state.app_df = None
        else:
            st.sidebar.error("❌ Data parsing failed. Check the error messages above.")
            st.session_state.data_processed = False
            st.session_state.app_df = None

        # Rerun to update the UI based on data_processed state
        st.rerun()

    # --- Application Modes ---
    app_mode = None
    if st.session_state.data_processed:
        st.sidebar.header("🚀 Analysis Modes")
        app_mode = st.sidebar.selectbox("Select Application Mode",
                                        ["🏦 Dormant Account Analyzer", "🔒 Compliance Analyzer", "🔍 SQL Bot",
                                         "💬 Chatbot Only"],
                                        key="app_mode_selector")

    st.title(f"{app_mode}" if app_mode else "Unified Banking Compliance Solution")

    # --- Display Processed Data ---
    if st.session_state.data_processed and st.session_state.app_df is not None:
        current_df = st.session_state.app_df.copy()
        st.header("Data Overview")
        if st.checkbox("View Processed Dataset (first 5 rows)", key="view_processed_data_checkbox"):
            display_df = current_df.head().copy()
            if 'column_mapping' in st.session_state and st.session_state['column_mapping']:
                try:
                    # Create a display mapping that only includes columns present in the current small dataframe
                    display_columns_mapping = {std_col: st.session_state['column_mapping'].get(std_col, std_col)
                                               for std_col in display_df.columns}
                    display_df.rename(columns=display_columns_mapping, inplace=True)
                    st.dataframe(display_df)
                    st.caption("Displaying original column names where available for the first 5 rows.")
                except Exception as e:
                    st.error(f"Error applying original column names for display: {e}")
                    st.dataframe(current_df.head())
                    st.caption("Displaying standardized column names for the first 5 rows.")
            else:
                st.dataframe(display_df)
                st.caption("Displaying standardized column names for the first 5 rows.")
        st.divider()

        # --- Mode Specific UI ---
        if app_mode == "🏦 Dormant Account Analyzer":
            st.subheader("🏦 Dormant Account Analysis")
            threshold = datetime.now() - timedelta(days=3 * 365) # 3 years inactivity threshold

            agent_option = st.selectbox("🧭 Choose Dormant Detection Agent", [
                "📊 Summarized Dormant Analysis", "🔐 Safe Deposit Box Agent", "💼 Investment Inactivity Agent",
                "🏦 Fixed Deposit Agent", "📉 3-Year General Inactivity Agent",
                "📵 Unreachable + No Active Accounts Agent"
            ], key="dormant_agent_selector")

            if agent_option == "📊 Summarized Dormant Analysis":
                st.subheader("📈 Summarized Dormant Analysis Results")
                if st.button("📊 Run Summarized Dormant Analysis", key="run_summary_dormant_button"):
                    # No need for dormant_summary_rerun flag with streamlit's rerun behavior
                    with st.spinner("Running all dormant checks..."):
                        sd_df, sd_count, sd_desc = check_safe_deposit(current_df, threshold)
                        inv_df, inv_count, inv_desc = check_investment_inactivity(current_df, threshold)
                        fd_df, fd_count, fd_desc = check_fixed_deposit_inactivity(current_df, threshold)
                        gen_df, gen_count, gen_desc = check_general_inactivity(current_df, threshold)
                        unr_df, unr_count, unr_desc = check_unreachable_dormant(current_df)
                        # Store results in session state if needed later (e.g., for PDF)
                        st.session_state.dormant_summary_results = {
                            "sd": {"df": sd_df, "count": sd_count, "desc": sd_desc},
                            "inv": {"df": inv_df, "count": inv_count, "desc": inv_desc},
                            "fd": {"df": fd_df, "count": fd_count, "desc": fd_desc},
                            "gen": {"df": gen_df, "count": gen_count, "desc": gen_desc},
                            "unr": {"df": unr_df, "count": unr_count, "desc": unr_desc},
                            "total_accounts": len(current_df)
                        }

                    results = st.session_state.dormant_summary_results
                    st.subheader("🔢 Numerical Summary")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Uncontacted Safe Deposit (>3y)", results["sd"]["count"], help=results["sd"]["desc"])
                        st.metric("General Inactivity (>3y)", results["gen"]["count"], help=results["gen"]["desc"])
                    with col2:
                        st.metric("Uncontacted Investment (>3y)", results["inv"]["count"], help=results["inv"]["desc"])
                        st.metric("Unreachable & 'Dormant'", results["unr"]["count"], help=results["unr"]["desc"])
                    with col3:
                        st.metric("Fixed Deposit Inactivity (>3y)", results["fd"]["count"], help=results["fd"]["desc"])

                    summary_input_text = (
                        f"Dormant Analysis Findings ({results['total_accounts']} total accounts analyzed, threshold >3 years inactive):\n"
                        f"- {results['sd']['desc']}\n- {results['inv']['desc']}\n- {results['fd']['desc']}\n- {results['gen']['desc']}\n- {results['unr']['desc']}")

                    st.subheader("📝 Narrative Summary")
                    if llm and LANGCHAIN_AVAILABLE:
                        try:
                            with st.spinner("Generating AI Summary..."):
                                # Improved summary prompt template
                                summary_prompt_template = PromptTemplate.from_template(
                                    """Act as a Senior Compliance Analyst AI. You have analyzed a dataset of banking accounts.
                                    Below is a numerical summary of accounts identified by specific dormant/inactivity criteria:

                                    {analysis_details}

                                    Based on these findings, provide a concise executive summary highlighting key risks, trends, and observations regarding dormant accounts. Keep it professional and focused on compliance implications.

                                    Executive Summary:"""
                                )
                                summary_chain = summary_prompt_template | llm | StrOutputParser()
                                narrative_summary = summary_chain.invoke({
                                    "analysis_details": summary_input_text
                                })
                                st.markdown(narrative_summary)
                                st.session_state.dormant_narrative_summary = narrative_summary # Store for PDF
                        except Exception as llm_e:
                            st.error(f"AI summary generation failed: {llm_e}")
                            st.text_area("Raw Findings:", summary_input_text, height=150)
                            st.session_state.dormant_narrative_summary = f"AI Summary Failed. Raw Findings:\n{summary_input_text}" # Store raw findings for PDF
                    else:
                        st.warning("AI Assistant not available. Cannot generate insights.")
                        st.text_area("Raw Findings:", summary_input_text, height=150)
                        st.session_state.dormant_narrative_summary = f"AI Not Available. Raw Findings:\n{summary_input_text}" # Store raw findings for PDF


                    st.subheader("⬇️ Export Summary")
                    if st.button("📄 Download Summary Report (PDF)", key="download_dormant_summary_pdf"):
                         pdf = FPDF(); pdf.add_page(); pdf.set_font("Arial", 'B', 14);
                         pdf.cell(0, 10, "Dormant Account Analysis Summary Report", 0, 1, 'C'); pdf.ln(5)
                         pdf.set_font("Arial", 'B', 12); pdf.cell(0, 10, "Numerical Summary", 0, 1);
                         pdf.set_font("Arial", size=10)
                         pdf.multi_cell(0, 6, f"- {results['sd']['desc']}".encode('latin-1', 'replace').decode('latin-1'))
                         pdf.multi_cell(0, 6, f"- {results['inv']['desc']}".encode('latin-1', 'replace').decode('latin-1'))
                         pdf.multi_cell(0, 6, f"- {results['fd']['desc']}".encode('latin-1', 'replace').decode('latin-1'))
                         pdf.multi_cell(0, 6, f"- {results['gen']['desc']}".encode('latin-1', 'replace').decode('latin-1'))
                         pdf.multi_cell(0, 6, f"- {results['unr']['desc']}".encode('latin-1', 'replace').decode('latin-1')); pdf.ln(5)

                         pdf.set_font("Arial", 'B', 12); pdf.cell(0, 10, "Narrative Summary (AI Generated or Raw Findings)", 0, 1);
                         pdf.set_font("Arial", size=10)
                         narrative_text = st.session_state.get('dormant_narrative_summary', "Summary not generated or AI failed.")
                         pdf.multi_cell(0, 6, narrative_text.encode('latin-1', 'replace').decode('latin-1')); pdf.ln(5)

                         pdf_file_name = f"dormant_summary_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                         try:
                            pdf_output = pdf.output(dest='S').encode('latin-1')
                            st.download_button(label="Click to Download PDF", data=pdf_output, file_name=pdf_file_name, mime="application/pdf")
                         except Exception as pdf_e:
                            st.error(f"Error generating PDF: {pdf_e}")

            # Individual dormant agent logic
            else:
                st.subheader(f"Agent Task Results: {agent_option}")
                data_filtered = pd.DataFrame()
                agent_desc = "Select an agent above."
                agent_executed = False

                agent_mapping = {
                    "🔐 Safe Deposit Box Agent": check_safe_deposit,
                    "💼 Investment Inactivity Agent": check_investment_inactivity,
                    "🏦 Fixed Deposit Agent": check_fixed_deposit_inactivity,
                    "📉 3-Year General Inactivity Agent": check_general_inactivity,
                    "📵 Unreachable + No Active Accounts Agent": check_unreachable_dormant
                }

                if selected_agent := agent_mapping.get(agent_option): # Use walrus operator for assignment and check
                    with st.spinner(f"Running {agent_option}..."):
                         # Pass necessary args based on agent
                         if agent_option in ["🔐 Safe Deposit Box Agent", "💼 Investment Inactivity Agent",
                                           "🏦 Fixed Deposit Agent", "📉 3-Year General Inactivity Agent"]:
                             data_filtered, count, agent_desc = selected_agent(current_df, threshold)
                         else: # "📵 Unreachable + No Active Accounts Agent"
                             data_filtered, count, agent_desc = selected_agent(current_df)
                         agent_executed = True

                    st.metric("Accounts Identified", count, help=agent_desc)
                    st.markdown(f"**Agent Description:** {agent_desc}")

                if agent_executed:
                     if not data_filtered.empty:
                         st.success(f"{len(data_filtered)} accounts identified.")
                         if st.checkbox(f"View first 15 detected accounts for '{agent_option}'", key=f"view_detected_{agent_option.replace(' ','_')}"):
                             st.dataframe(data_filtered.head(15))

                         if llm and LANGCHAIN_AVAILABLE:
                             sample_size = min(15, len(data_filtered))
                             sample_data_csv = data_filtered.sample(n=sample_size).to_csv(index=False)

                             # Insight generation prompts
                             observation_prompt = PromptTemplate.from_template(
                                 "You are a senior bank analyst. Analyze the following sample data from identified dormant/inactive accounts and provide only key observations about the dataset structure, common values, or patterns.\n\nSample Data (CSV):\n{data}\n\nObservations:")
                             trend_prompt = PromptTemplate.from_template(
                                 "You are a data strategist. Given the following sample data from a banking compliance analysis, identify potential trends or significant findings relevant to compliance or risk.\n\nSample Data (CSV):\n{data}\n\nTrends and Findings:")
                             narration_prompt = PromptTemplate.from_template(
                                 "You are writing a CXO summary based on compliance analysis. Using the provided observations and trends, craft a concise executive summary (max 3-4 sentences) suitable for a busy executive.\n\nObservations:\n{observation}\n\nTrends:\n{trend}\n\nExecutive Summary:")
                             action_prompt = PromptTemplate.from_template(
                                 "You are a strategic advisor to a bank's compliance department. Based on the following observations and trends, suggest specific, actionable steps the bank should take to address the identified issues.\n\nObservations:\n{observation}\n\nTrends:\n{trend}\n\nRecommended Actions:")

                             output_parser = StrOutputParser()
                             observation_chain = observation_prompt | llm | output_parser
                             trend_chain = trend_prompt | llm | output_parser

                             if st.button(f"Generate Insights for '{agent_option}'", key=f"generate_insights_{agent_option.replace(' ','_')}"):
                                 with st.spinner("Running insight agents..."):
                                     obs_output = observation_chain.invoke({"data": sample_data_csv})
                                     trend_output = trend_chain.invoke({"data": sample_data_csv})
                                     narration_chain = narration_prompt | llm | output_parser
                                     action_chain = action_prompt | llm | output_parser
                                     final_insight = narration_chain.invoke({"observation": obs_output, "trend": trend_output})
                                     action_output = action_chain.invoke({"observation": obs_output, "trend": trend_output})

                                 # Store insights in session state for display and PDF
                                 st.session_state[f'{agent_option}_insights'] = {
                                     'observation': obs_output,
                                     'trend': trend_output,
                                     'summary': final_insight,
                                     'actions': action_output
                                 }

                                 # Save to DB log
                                 if db_initialized:
                                     if save_summary_to_db(obs_output, trend_output, final_insight, action_output): # Uses default connection
                                         st.success("Insights saved to insight log.")
                                     else:
                                         st.error("Failed to save insights to DB log.")
                                 else:
                                     st.warning("DB not initialized. Insights not saved to log.")

                             # Display insights if they exist in session state
                             if f'{agent_option}_insights' in st.session_state:
                                 insights = st.session_state[f'{agent_option}_insights']
                                 with st.expander("🔍 Observation Insight"):
                                     st.markdown(insights['observation'])
                                 with st.expander("📊 Trend Insight"):
                                     st.markdown(insights['trend'])
                                 with st.expander("📌 CXO Summary"):
                                     st.markdown(insights['summary'])
                                 with st.expander("🚀 Recommended Actions"):
                                     st.markdown(insights['actions'])

                                 # PDF export for individual agent insights
                                 if st.button(f"📄 Download '{agent_option}' Report (PDF)", key=f"download_pdf_{agent_option.replace(' ','_')}"):
                                    pdf = FPDF(); pdf.add_page(); pdf.set_font("Arial", 'B', 14);
                                    pdf.cell(0, 10, f"{agent_option} - Analysis Report", 0, 1, 'C'); pdf.ln(5)

                                    # Add obs_output, trend_output, final_insight, action_output to PDF
                                    pdf.set_font("Arial", 'B', 12); pdf.cell(0, 10, "Observations", 0, 1);
                                    pdf.set_font("Arial", size=10); pdf.multi_cell(0, 6, insights['observation'].encode('latin-1', 'replace').decode('latin-1')); pdf.ln(5)

                                    pdf.set_font("Arial", 'B', 12); pdf.cell(0, 10, "Trends", 0, 1);
                                    pdf.set_font("Arial", size=10); pdf.multi_cell(0, 6, insights['trend'].encode('latin-1', 'replace').decode('latin-1')); pdf.ln(5)

                                    pdf.set_font("Arial", 'B', 12); pdf.cell(0, 10, "Executive Summary", 0, 1);
                                    pdf.set_font("Arial", size=10); pdf.multi_cell(0, 6, insights['summary'].encode('latin-1', 'replace').decode('latin-1')); pdf.ln(5)

                                    pdf.set_font("Arial", 'B', 12); pdf.cell(0, 10, "Recommended Actions", 0, 1);
                                    pdf.set_font("Arial", size=10); pdf.multi_cell(0, 6, insights['actions'].encode('latin-1', 'replace').decode('latin-1'))

                                    pdf_file_name_agent = f"{agent_option.replace(' ', '_').replace(':', '')}_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                                    try:
                                        pdf_output_agent = pdf.output(dest='S').encode('latin-1')
                                        st.download_button(label="Click to Download PDF", data=pdf_output_agent,
                                                           file_name=pdf_file_name_agent, mime="application/pdf",
                                                           key=f"download_pdf_button_{agent_option.replace(' ','_')}") # Unique key
                                    except Exception as pdf_e_agent:
                                        st.error(f"Error generating PDF: {pdf_e_agent}")

                         elif len(data_filtered) > 0:
                              st.info(f"AI Assistant not available to generate insights.")

                     elif len(data_filtered) == 0:
                         st.info("No accounts matching the criteria were found.")


        elif app_mode == "🔒 Compliance Analyzer":
            st.subheader("🔒 Compliance Analysis Tasks")

            agent_options_compliance = ["📊 Summarized Compliance Detection", "📨 Contact Attempt Verification Agent",
                             "🚩 Flag Dormant Candidate Agent",
                             "📘 Dormant Ledger Review Agent", "❄️ Account Freeze Candidate Agent",
                             "🏦 CBUAE Transfer Candidate Agent"]
            selected_agent_compliance = st.selectbox("Select Compliance Task or Summary", agent_options_compliance,
                                          key="compliance_agent_selector")

            # Thresholds for compliance checks
            st.sidebar.subheader("Compliance Thresholds")
            general_inactivity_threshold_days = st.sidebar.number_input("Flagging Inactivity Threshold (days)", min_value=1,
                                                               value=3 * 365, step=30, key="flag_threshold_days") # Default 3 years
            general_inactivity_threshold_date = datetime.now() - timedelta(days=general_inactivity_threshold_days)
            st.sidebar.caption(f"Flagging Threshold: {general_inactivity_threshold_date.strftime('%Y-%m-%d')}")

            freeze_inactivity_threshold_days = st.sidebar.number_input("Freeze Inactivity Threshold (days)", min_value=1,
                                                               value=5 * 365, step=30, key="freeze_threshold_days") # Default 5 years
            freeze_inactivity_threshold_date = datetime.now() - timedelta(days=freeze_inactivity_threshold_days)
            st.sidebar.caption(f"Freeze Threshold: {freeze_inactivity_threshold_date.strftime('%Y-%m-%d')}")


            default_cbuae_date_str = "2020-04-24" # Example CBUAE date
            cbuae_cutoff_str = st.sidebar.text_input("CBUAE Transfer Cutoff Date (YYYY-MM-DD)", value=default_cbuae_date_str,
                                             key="cbuae_cutoff_date")
            cbuae_cutoff_date = None
            try:
                cbuae_cutoff_date = datetime.strptime(cbuae_cutoff_str, "%Y-%m-%d"); st.sidebar.caption(
                    f"Using CBUAE cutoff: {cbuae_cutoff_date.strftime('%Y-%m-%d')}")
            except ValueError:
                st.sidebar.error("Invalid CBUAE cutoff date format. Transfer agent will be skipped.")


            if selected_agent_compliance == "📊 Summarized Compliance Detection":
                st.subheader("📈 Summarized Compliance Detection Results")
                # Summarized logic - run all and show counts/summaries
                if st.button("📊 Run Summarized Compliance Analysis", key="run_summary_compliance_button"):
                    with st.spinner("Running all compliance checks..."):
                         contact_df, contact_count, contact_desc = detect_incomplete_contact(current_df)
                         flag_df, flag_count, flag_desc = detect_flag_candidates(current_df, general_inactivity_threshold_date)
                         ledger_df, ledger_count, ledger_desc = detect_ledger_candidates(current_df)
                         freeze_df, freeze_count, freeze_desc = detect_freeze_candidates(current_df, freeze_inactivity_threshold_date)
                         transfer_df, transfer_count, transfer_desc = detect_transfer_candidates(current_df, cbuae_cutoff_date)

                         st.session_state.compliance_summary_results = {
                            "contact": {"df": contact_df, "count": contact_count, "desc": contact_desc},
                            "flag": {"df": flag_df, "count": flag_count, "desc": flag_desc},
                            "ledger": {"df": ledger_df, "count": ledger_count, "desc": ledger_desc},
                            "freeze": {"df": freeze_df, "count": freeze_count, "desc": freeze_desc},
                            "transfer": {"df": transfer_df, "count": transfer_count, "desc": transfer_desc},
                            "total_accounts": len(current_df)
                        }

                    results = st.session_state.compliance_summary_results
                    st.subheader("🔢 Numerical Summary")
                    st.metric("Incomplete Contact Attempts", results['contact']['count'], help=results['contact']['desc'])
                    st.metric(f"Flag Candidates (>={general_inactivity_threshold_days} days inactive)", results['flag']['count'], help=results['flag']['desc'])
                    st.metric("Ledger Classification Needed", results['ledger']['count'], help=results['ledger']['desc'])
                    st.metric(f"Freeze Candidates (>={freeze_inactivity_threshold_days} days dormant)", results['freeze']['count'], help=results['freeze']['desc'])
                    st.metric(f"CBUAE Transfer Candidates (Inactive before {cbuae_cutoff_str})", results['transfer']['count'], help=results['transfer']['desc'])

                    # Add AI narrative summary similar to dormant analysis if LLM is available
                    if llm and LANGCHAIN_AVAILABLE:
                        compliance_summary_input_text = (
                            f"Compliance Analysis Findings ({results['total_accounts']} total accounts analyzed):\n"
                            f"- {results['contact']['desc']}\n"
                            f"- {results['flag']['desc']}\n"
                            f"- {results['ledger']['desc']}\n"
                            f"- {results['freeze']['desc']}\n"
                            f"- {results['transfer']['desc']}"
                        )
                        st.subheader("📝 AI Compliance Summary")
                        try:
                            with st.spinner("Generating AI Compliance Summary..."):
                                compliance_summary_prompt_template = PromptTemplate.from_template(
                                    """Act as a Senior Banking Compliance Officer AI. You have reviewed the output of several compliance agents.
                                    Here are the counts and descriptions of accounts identified by each compliance check:

                                    {compliance_details}

                                    Provide a brief executive summary highlighting the most significant compliance risks and areas requiring immediate attention based on these findings. Focus on actionable insights for the compliance team.

                                    Compliance Summary:"""
                                )
                                compliance_summary_chain = compliance_summary_prompt_template | llm | StrOutputParser()
                                compliance_narrative_summary = compliance_summary_chain.invoke({
                                    "compliance_details": compliance_summary_input_text
                                })
                            st.markdown(compliance_narrative_summary)
                            st.session_state.compliance_narrative_summary = compliance_narrative_summary # Store for PDF
                        except Exception as llm_e:
                            st.error(f"AI compliance summary generation failed: {llm_e}")
                            st.text_area("Raw Compliance Findings:", compliance_summary_input_text, height=150)
                            st.session_state.compliance_narrative_summary = f"AI Summary Failed. Raw Findings:\n{compliance_summary_input_text}"
                    else:
                        st.warning("AI Assistant not available to generate a summary.")

                    # Add PDF download for compliance summary (similar to dormant)
                    st.subheader("⬇️ Export Summary")
                    if st.button("📄 Download Compliance Summary Report (PDF)", key="download_compliance_summary_pdf"):
                         pdf = FPDF(); pdf.add_page(); pdf.set_font("Arial", 'B', 14);
                         pdf.cell(0, 10, "Compliance Analysis Summary Report", 0, 1, 'C'); pdf.ln(5)
                         pdf.set_font("Arial", 'B', 12); pdf.cell(0, 10, "Numerical Summary", 0, 1);
                         pdf.set_font("Arial", size=10)
                         pdf.multi_cell(0, 6, results['contact']['desc'].encode('latin-1', 'replace').decode('latin-1'))
                         pdf.multi_cell(0, 6, results['flag']['desc'].encode('latin-1', 'replace').decode('latin-1'))
                         pdf.multi_cell(0, 6, results['ledger']['desc'].encode('latin-1', 'replace').decode('latin-1'))
                         pdf.multi_cell(0, 6, results['freeze']['desc'].encode('latin-1', 'replace').decode('latin-1'))
                         pdf.multi_cell(0, 6, results['transfer']['desc'].encode('latin-1', 'replace').decode('latin-1')); pdf.ln(5)

                         pdf.set_font("Arial", 'B', 12); pdf.cell(0, 10, "Narrative Summary (AI Generated or Raw Findings)", 0, 1);
                         pdf.set_font("Arial", size=10)
                         narrative_text = st.session_state.get('compliance_narrative_summary', "Summary not generated or AI failed.")
                         pdf.multi_cell(0, 6, narrative_text.encode('latin-1', 'replace').decode('latin-1')); pdf.ln(5)

                         pdf_file_name = f"compliance_summary_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                         try:
                            pdf_output = pdf.output(dest='S').encode('latin-1')
                            st.download_button(label="Click to Download PDF", data=pdf_output, file_name=pdf_file_name, mime="application/pdf")
                         except Exception as pdf_e:
                            st.error(f"Error generating PDF: {pdf_e}")


            else: # Individual Compliance Agent Logic
                st.subheader(f"Agent Task Results: {selected_agent_compliance}")
                data_filtered = pd.DataFrame()
                agent_desc = "Select an agent above."
                agent_executed = False

                agent_mapping_compliance = {
                    "📨 Contact Attempt Verification Agent": detect_incomplete_contact,
                    "🚩 Flag Dormant Candidate Agent": detect_flag_candidates,
                    "📘 Dormant Ledger Review Agent": detect_ledger_candidates,
                    "❄️ Account Freeze Candidate Agent": detect_freeze_candidates,
                    "🏦 CBUAE Transfer Candidate Agent": detect_transfer_candidates
                }

                if selected_agent_func := agent_mapping_compliance.get(selected_agent_compliance): # Use walrus operator
                     with st.spinner(f"Running {selected_agent_compliance}..."):
                         # Pass necessary args based on agent
                         if selected_agent_compliance == "🚩 Flag Dormant Candidate Agent":
                             data_filtered, count, agent_desc = selected_agent_func(current_df, general_inactivity_threshold_date)
                         elif selected_agent_compliance == "❄️ Account Freeze Candidate Agent":
                             data_filtered, count, agent_desc = selected_agent_func(current_df, freeze_inactivity_threshold_date)
                         elif selected_agent_compliance == "🏦 CBUAE Transfer Candidate Agent":
                             # Handle potential invalid date from sidebar input
                             if cbuae_cutoff_date is None:
                                 data_filtered, count, agent_desc = pd.DataFrame(), 0, "Skipped due to invalid CBUAE cutoff date format."
                             else:
                                 data_filtered, count, agent_desc = selected_agent_func(current_df, cbuae_cutoff_date)
                         else: # "📨 Contact Attempt Verification Agent", "📘 Dormant Ledger Review Agent"
                             data_filtered, count, agent_desc = selected_agent_func(current_df)
                         agent_executed = True
                         st.metric("Accounts Identified", count, help=agent_desc)

                if agent_executed:
                     if not data_filtered.empty:
                         st.success(f"{len(data_filtered)} accounts identified.")
                         if st.checkbox(f"View first 15 detected accounts for '{selected_agent_compliance}'", key=f"view_detected_{selected_agent_compliance.replace(' ','_')}"):
                            st.dataframe(data_filtered.head(15))

                         # Add buttons for next steps/logging based on agent type
                         if selected_agent_compliance == "🚩 Flag Dormant Candidate Agent":
                            if st.button("Log Flagging Instruction to DB (for Audit)", key="log_flag_instruction_compliance"):
                                if db_initialized:
                                    conn_flag = get_db_connection() # Uses cached connection
                                    if conn_flag:
                                        try:
                                            with conn_flag:
                                                cursor_flag = conn_flag.cursor()
                                                # Use Account_ID column explicitly
                                                if 'Account_ID' in data_filtered.columns:
                                                    flagged_ids = data_filtered['Account_ID'].tolist()
                                                    timestamp_now = datetime.now()
                                                    # Check if dormant_flags table exists before inserting
                                                    cursor_flag.execute("SELECT 1 FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'dormant_flags'")
                                                    table_exists = cursor_flag.fetchone() is not None
                                                    if table_exists:
                                                        # Only insert if account_id does not already exist
                                                        insert_sql = """
                                                            INSERT INTO dormant_flags(account_id, flag_instruction, timestamp)
                                                            SELECT ?, ?, ?
                                                            WHERE NOT EXISTS (SELECT 1 FROM dormant_flags WHERE account_id = ?)
                                                        """
                                                        rows_inserted = 0
                                                        for acc_id in flagged_ids:
                                                             # Ensure account_id is not None/empty string
                                                             if pd.notna(acc_id) and str(acc_id).strip() != '':
                                                                 cursor_flag.execute(insert_sql, (str(acc_id), f"Identified by {selected_agent_compliance} for review (Threshold: {general_inactivity_threshold_days} days)", timestamp_now, str(acc_id)))
                                                                 rows_inserted += cursor_flag.rowcount # Count how many rows were actually inserted
                                                            # Removed individual commit, commit outside the loop
                                                        conn_flag.commit()
                                                        st.success(f"Logged {rows_inserted} unique accounts for flagging review!")
                                                        if rows_inserted < len(flagged_ids):
                                                            st.info(f"Note: {len(flagged_ids) - rows_inserted} accounts were already in the flagging log.")
                                                    else:
                                                        st.error("Error: 'dormant_flags' table not found in the database. Cannot log.")
                                                else:
                                                     st.error("DataFrame does not have 'Account_ID' column. Cannot log.")
                                        except Exception as db_e:
                                            st.error(f"DB logging failed: {db_e}")
                                    else:
                                        st.error("Failed to connect to DB for logging.")
                                else:
                                     st.warning("DB not initialized. Cannot log flagging instruction.")

                         # Add other actions for Ledger, Freeze, Transfer candidates as needed
                         if selected_agent_compliance == "📘 Dormant Ledger Review Agent":
                              st.info("Review the accounts identified for manual classification in the dormant ledger.")
                              # Maybe add a button to export this list to CSV
                              if not data_filtered.empty and st.button(f"Download Ledger Candidates CSV", key=f"download_ledger_candidates_csv"):
                                  csv_data = data_filtered.to_csv(index=False).encode('utf-8')
                                  st.download_button(label="Click to Download CSV", data=csv_data, file_name=f"ledger_candidates_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv")

                         if selected_agent_compliance in ["❄️ Account Freeze Candidate Agent", "🏦 CBUAE Transfer Candidate Agent"]:
                             st.info("Accounts identified for potential freeze or transfer based on regulations.")
                             if not data_filtered.empty and st.button(f"Download {selected_agent_compliance.split(' ')[-2]} Candidates CSV", key=f"download_{selected_agent_compliance.replace(' ','_')}_csv"):
                                  csv_data = data_filtered.to_csv(index=False).encode('utf-8')
                                  st.download_button(label="Click to Download CSV", data=csv_data, file_name=f"{selected_agent_compliance.replace(' ','_').replace('Agent','candidates').lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv")


                     elif len(data_filtered) == 0:
                         st.info("No accounts matching the criteria were found.")

        elif app_mode == "🔍 SQL Bot":
            st.header("SQL Database Query Bot")
            # --- Explicitly state which database is being queried ---
            # Use the DB_SERVER and DB_NAME constants which are sourced from env/secrets
            st.info(f"🤖 This bot queries the **default database**: `{DB_NAME}` on server `{DB_SERVER}` as configured via secrets/environment variables.")
            st.caption("*(This is separate from the 'Load Data' feature, which brings data into the app's memory.)*") # Added clarifying caption


            if not db_initialized:
                st.warning("Cannot use SQL Bot: Default database connection failed during initialization.")
            elif not llm or not LANGCHAIN_AVAILABLE:
                st.warning(
                    "AI Assistant (Groq/Langchain) needed for NL to SQL. Configure API key and install libraries.")
            else:
                # Fetch schema for the default database
                schema = get_db_schema()  # This function now ONLY uses get_db_connection() (cached connection)

                if schema:
                    schema_text = "Database Schema for SQL Bot:\n"
                    for table, columns_list in schema.items():
                        schema_text += f"Table: {table}\nColumns:\n{chr(10).join([f'- {name} ({dtype})' for name, dtype in columns_list])}\n\n"
                    with st.expander("Show Database Schema (from default DB)"):
                        st.code(schema_text, language='text')

                    # SQL Bot NL to SQL generation and execution
                    nl_query_sqlbot = st.text_area("Ask a database question:",
                                                   placeholder="e.g., How many dormant accounts in 'Dubai' branch from the 'accounts_data' table?",
                                                   height=100, key="sql_bot_nl_query_input")

                    generate_execute_sqlbot = st.button("Generate & Execute SQL Query",
                                                        key="sql_bot_generate_execute_button")

                    # --- Initialize sql_query_generated before the try block ---
                    sql_query_generated = None
                    # --- End of Initialization ---

                    if generate_execute_sqlbot and nl_query_sqlbot:

                        # --- Generate SQL ---
                        nl_to_sql_prompt = PromptTemplate.from_template(
                            """You are an expert Azure SQL query generator. Given the database schema and a user question in natural language, generate *only* the valid T-SQL query that answers the question.
                            Adhere strictly to the schema provided. Only use tables and columns exactly as they are named in the schema.
                            The database is Azure SQL Server. Ensure the query syntax is correct for T-SQL.
                            Prioritize using the 'accounts_data' table for general account questions. Use 'dormant_flags' or 'sql_query_history' if the question is specifically about those logs.
                            The user expects a query to *retrieve* data. Generate *only* SELECT statements. Do NOT generate INSERT, UPDATE, DELETE, CREATE, ALTER, or DROP statements, or any other SQL commands.
                            Do NOT include any explanations, greetings, or markdown code block formatting (```sql```) around the query. Just output the plain SQL query text.
                            If the question cannot be answered using *only* the provided schema and *only* a SELECT query, output a polite message stating that you cannot answer that query based on the schema.

                            Database Schema (Azure SQL):
                            {schema}

                            User question: {question}

                            T-SQL Query:"""
                        )

                        nl_to_sql_chain = nl_to_sql_prompt | llm | StrOutputParser()

                        try:
                            with st.spinner("🤖 Converting NL to SQL..."):
                                sql_query_raw = nl_to_sql_chain.invoke(
                                    {"schema": schema_text, "question": nl_query_sqlbot.strip()})

                                # More aggressive cleaning to remove potential leading/trailing text around the query
                                # This regex tries to find the first SELECT and everything after it, up to the end or a common trailing pattern
                                match = re.search(r"SELECT.*", sql_query_raw, re.IGNORECASE | re.DOTALL)
                                if match:
                                    sql_query_generated = match.group(0).strip()
                                else:
                                    # Fallback to original cleaning if SELECT is not found explicitly by regex
                                    # Only use this fallback if the regex failed
                                    sql_query_generated_fallback = re.sub(r"^```sql\s*|\s*```$", "", sql_query_raw,
                                                                          flags=re.MULTILINE).strip()
                                    st.warning("Could not find SELECT clearly with regex. Using fallback cleaning.")
                                    sql_query_generated = sql_query_generated_fallback  # Assign fallback result

                            # Validate the generated query is a SELECT statement (basic check)
                            if not sql_query_generated or not sql_query_generated.lower().strip().startswith(
                                    "select"):  # Added check for empty/None
                                st.error(
                                    "Generated text does not start with SELECT or is empty. It cannot be executed. Raw output might contain unexpected text.")
                                # Optionally display the raw output for debugging
                                # st.code(sql_query_raw, language='text')
                                sql_query_generated = None  # Explicitly set back to None if validation fails

                        except Exception as e:
                            st.error(f"SQL generation error: {e}")
                            sql_query_generated = None  # Ensure it's None if *any* error occurs during generation

                        # --- Execute SQL (if generated successfully) ---
                        # This block is now correctly guarded by `if sql_query_generated:`
                        if sql_query_generated:
                            # ... (display code, execute query, save history) ...
                            st.subheader("Generated SQL Query")
                            st.code(sql_query_generated, language='sql')

                            conn_for_exec = get_db_connection()  # Use the default cached connection for execution
                            if conn_for_exec is None:
                                st.error("Cannot execute query: Default DB connection is not available.")
                            else:
                                try:
                                    with st.spinner("⏳ Executing query..."):
                                        # Use pd.read_sql which takes a connection object
                                        results_df = pd.read_sql(sql_query_generated, conn_for_exec)

                                    # Save query to history (uses the same default DB connection)
                                    if db_initialized:  # Only try to save if DB init was successful
                                        try:
                                            # No need for a separate connection object here, reuse conn_for_exec
                                            with conn_for_exec.cursor() as cursor_hist_exec:
                                                # Check if table exists before inserting
                                                cursor_hist_exec.execute(
                                                    "SELECT 1 FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'sql_query_history'")
                                                history_table_exists = cursor_hist_exec.fetchone() is not None
                                                if history_table_exists:
                                                    cursor_hist_exec.execute(
                                                        "INSERT INTO sql_query_history (natural_language_query, sql_query) VALUES (?, ?)",
                                                        (nl_query_sqlbot, sql_query_generated))
                                                    conn_for_exec.commit()  # Commit history save
                                                    st.success("Query saved to history.")
                                                else:
                                                    st.warning("SQL query history table not found. Query not saved.")
                                        except Exception as history_e:
                                            st.warning(f"Failed to save query to history: {history_e}")
                                    else:
                                        st.warning("DB not initialized. Query not saved to history.")

                                    st.subheader("Query Results")
                                    if not results_df.empty:
                                        st.dataframe(results_df)
                                        st.info(f"Query returned {len(results_df)} rows.")
                                        # CSV download button for results
                                        csv_data = results_df.to_csv(index=False).encode('utf-8')
                                        st.download_button(label="Download Results as CSV", data=csv_data,
                                                           file_name=f"sql_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                                           mime="text/csv")
                                    else:
                                        st.info("Query executed successfully but returned no results.")

                                except Exception as e:
                                    st.error(f"Query execution error: {e}")
                                finally:
                                    # NO LONGER close conn_for_exec here. It's the cached connection.
                                    pass


                    # Store generated SQL in session state *after* the generation and execution attempt
                    # This line is now safe because sql_query_generated is always initialized to None
                    # or assigned a value (either the generated query or None again on validation/error)
                    st.session_state.last_nl_query_sqlbot = nl_query_sqlbot  # Store NL query
                    st.session_state.generated_sql_sqlbot = sql_query_generated  # Store generated SQL

                    # Add an option to get an explanation for the *last generated* SQL query
                    # This button is now also safely guarded by checking session state
                    if st.session_state.get(
                            'generated_sql_sqlbot') and st.session_state.generated_sql_sqlbot:  # Check if generated_sql is not None/empty
                        if st.button("Explain Generated Query", key="analyze_sql_button_sqlbot"):
                            if llm and LANGCHAIN_AVAILABLE:
                                with st.spinner("🧠 Analyzing query..."):
                                    sql_explanation_prompt = PromptTemplate.from_template(
                                        """You are a data analyst explaining an SQL query. Provide a clear, concise explanation of what the following T-SQL query does, referencing the provided database schema.

                                        Database Schema:
                                        {schema}

                                        SQL Query:
                                        ```sql
                                        {sql_query}
                                        ```

                                        Explanation:"""
                                    )
                                    sql_explanation_chain = sql_explanation_prompt | llm | StrOutputParser()
                                    try:
                                        explanation = sql_explanation_chain.invoke(
                                            {"sql_query": st.session_state.generated_sql_sqlbot,
                                             "schema": schema_text})
                                        st.subheader("Query Analysis")
                                        st.markdown(explanation)
                                    except Exception as e:
                                        st.error(f"Query explanation error: {e}")
                            else:
                                st.warning("AI Assistant (LLM) needed for query explanation.")


                else:
                    st.warning("Could not retrieve database schema from the default DB. SQL Bot is limited.")


            st.subheader("Query History")
            # Show history from default DB's history table
            if st.checkbox("Show Recent SQL Queries from History", key="show_sql_history_checkbox_sqlbot"):
                if db_initialized:
                    conn_hist_disp = get_db_connection() # Use default DB connection (cached) for history
                    if conn_hist_disp:
                        try:
                            # Use pd.read_sql with the connection object
                            history_df = pd.read_sql(
                                "SELECT TOP 10 timestamp, natural_language_query, sql_query FROM sql_query_history ORDER BY timestamp DESC",
                                conn_hist_disp) # Use TOP for SQL Server
                            # DO NOT close conn_hist_disp here.
                            if not history_df.empty:
                                for _, row in history_df.iterrows():
                                    ts = pd.to_datetime(row['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
                                    with st.expander(
                                        f"Query at {ts}: \"{row['natural_language_query'][:70]}...\""): # Truncate NL query for title
                                            st.text_area("Natural Language:", row['natural_language_query'], height=70, disabled=True)
                                            st.code(row['sql_query'], language='sql')
                                    st.markdown("---") # Separator
                            else:
                                st.info("No queries in history yet.")
                        except Exception as e:
                            st.error(f"Error retrieving query history: {e}")
                        finally:
                             # NO LONGER close conn_hist_disp here.
                             pass
                    else:
                        st.info("Cannot connect to default DB to show history.")
                else:
                     st.info("DB not initialized. Cannot show history.")


        elif app_mode == "💬 Chatbot Only":
            st.header("Banking Compliance Chatbot")
            st.info("💬 Ask questions or request plots about the **loaded data**.")

            # Display chat messages
            for message in st.session_state.chat_messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    # Check if a chart object exists and display it
                    if "chart" in message and message["chart"] is not None:
                         try:
                            st.plotly_chart(message["chart"], use_container_width=True)
                         except Exception as e:
                             st.warning(f"Could not display chart: {e}")


            # Chat input
            prompt_chat = st.chat_input("Ask a question about the loaded data (e.g., 'Show me a bar chart of account types', 'How many dormant accounts?')...")

            if prompt_chat:
                # Add user message to chat history
                st.session_state.chat_messages.append({"role": "user", "content": prompt_chat})
                # Display user message
                with st.chat_message("user"):
                    st.markdown(prompt_chat)

                # Get and display assistant response (text + optional chart)
                with st.chat_message("assistant"):
                    # Use current_df (copy of st.session_state.app_df) for the bot
                    response_text_chat, chart_obj_chat = get_response_and_chart(prompt_chat, current_df, llm)
                    st.markdown(response_text_chat)
                    if chart_obj_chat is not None:
                         try:
                            st.plotly_chart(chart_obj_chat, use_container_width=True)
                         except Exception as e:
                              st.warning(f"Could not display generated chart: {e}")

                # Add assistant response (and chart) to chat history
                assistant_response_chat = {"role": "assistant", "content": response_text_chat}
                if chart_obj_chat is not None:
                    assistant_response_chat["chart"] = chart_obj_chat # Store the chart object
                st.session_state.chat_messages.append(assistant_response_chat)


    else: # No data processed
        st.info("👆 Please upload or load data using the sidebar options and click 'Process' to begin analysis.")
        st.header("Getting Started")
        st.markdown("""
        Welcome to the Unified Banking Compliance Solution.
        This application helps you analyze banking account data for compliance purposes, particularly focusing on dormant accounts.

        **Steps:**
        1.  **Upload Data:** Use the sidebar to upload your account data via CSV, XLSX, JSON, or fetch directly from a URL or an Azure SQL Database table.
        2.  **Process Data:** Click the "Process Uploaded/Fetched Data" button. The app will standardize column names and attempt to save the data to the configured default Azure SQL Database.
        3.  **Select Mode:** Once data is processed, choose an analysis mode from the sidebar:
            *   **Dormant Account Analyzer:** Run pre-defined agents to identify different categories of potentially dormant or high-risk accounts.
            *   **Compliance Analyzer:** Run compliance checks (e.g., contact verification, flagging candidates, ledger review, freeze/transfer candidates).
            *   **SQL Bot:** Query the **default** database (where processed data is saved) using natural language (requires AI Assistant).
            *   **Chatbot Only:** Ask questions or request simple visualizations about the **loaded dataset** using natural language (requires AI Assistant).

        **Configuration:**
        *   Database connection and AI features require credentials stored in `.streamlit/secrets.toml` or set as environment variables (`DB_USERNAME`, `DB_PASSWORD`, `GROQ_API_KEY`, etc.).
        *   The default Azure SQL server (`DB_SERVER`) and database (`DB_NAME`) constants can be overridden via environment variables.
        *   Ensure your Azure SQL server firewall allows connections from the IP address where you are running this application.
        """)
        st.markdown("---")
        st.markdown("Developed as a demonstration of AI-powered compliance tools.")


# === SQL Bot Helper (Fetches schema from the *default* DB) ===
@st.cache_data(show_spinner="Fetching database schema for SQL Bot...", ttl="1h") # Cache schema for 1 hour
def get_db_schema():
    """Fetches schema for the default database."""
    conn = get_db_connection() # Always use the default cached connection helper

    if conn is None:
        # Error message is already shown by get_db_connection
        return None

    schema_info = {}
    db_identifier = f"default database '{DB_NAME}' on '{DB_SERVER}'"

    try:
        with conn: # Use context manager with the cached connection
            cursor = conn.cursor()
            # Fetch tables from the default database
            cursor.execute("SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE = 'BASE TABLE'")
            tables = cursor.fetchall()
            for table_row in tables:
                table_name = table_row[0]
                # Fetch columns for each table
                cursor.execute(
                    f"SELECT COLUMN_NAME, DATA_TYPE, CHARACTER_MAXIMUM_LENGTH, IS_NULLABLE FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = ?", (table_name,))
                columns_raw = cursor.fetchall()
                column_details = []
                for col_raw in columns_raw:
                    col_name, data_type, max_length, is_nullable = col_raw[0], col_raw[1], col_raw[2], col_raw[3]
                    type_info = f"{data_type}"
                    if data_type in ('varchar', 'nvarchar', 'char', 'nchar', 'binary', 'varbinary'):
                        if max_length == -1:
                             type_info += "(MAX)"
                        elif max_length is not None:
                             type_info += f"({max_length})"
                    elif data_type in ('decimal', 'numeric'):
                         # Fetch precision and scale
                         try:
                            cursor.execute(f"SELECT NUMERIC_PRECISION, NUMERIC_SCALE FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = ? AND COLUMN_NAME = ?", (table_name, col_name))
                            prec_scale = cursor.fetchone()
                            if prec_scale:
                                precision, scale = prec_scale
                                type_info += f"({precision},{scale})"
                         except Exception as e:
                             # print(f"Warning: Could not fetch precision/scale for {table_name}.{col_name}: {e}")
                             pass # Ignore if fetching precision/scale fails
                    elif data_type in ('float', 'real'):
                        # Fetch precision for float/real
                        try:
                            cursor.execute(f"SELECT NUMERIC_PRECISION FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = ? AND COLUMN_NAME = ?", (table_name, col_name))
                            precision = cursor.fetchone()
                            if precision and precision[0] is not None:
                                 type_info += f"({precision[0]})"
                        except Exception as e:
                             # print(f"Warning: Could not fetch precision for float {table_name}.{col_name}: {e}")
                             pass # Ignore

                    nullable_status = "NULL" if is_nullable == "YES" else "NOT NULL"

                    column_details.append((col_name, f"{type_info} {nullable_status}")) # Add nullable status to schema info

                schema_info[table_name] = column_details

        st.sidebar.success("✅ Database schema fetched.")
        return schema_info

    except pyodbc.Error as e:
        st.error(f"SQL Bot: Database error fetching schema from {db_identifier}: {e}")
        return None
    except Exception as e:
        st.error(f"SQL Bot: Unexpected error fetching schema from {db_identifier}: {e}")
        return None


# Additional helper to test database connection
def test_db_connection(connection_string, display_area="sidebar"):
    """
    Tests a database connection string and returns diagnostic information.

    Args:
        connection_string: The connection string to test (password will be masked in output)
        display_area: Where to display messages ("sidebar" or "main")

    Returns:
        True if connection succeeded, False otherwise
    """
    # Mask the password in the connection string for display
    masked_conn_str = re.sub(r"PWD=[^;]*", "PWD=*****", connection_string)

    display_func = st.sidebar if display_area == "sidebar" else st

    try:
        display_func.info(f"Testing connection with: {masked_conn_str}")
        start_time = time.time()
        connection = pyodbc.connect(connection_string, timeout=30)
        end_time = time.time()

        # Test if we can actually execute a simple query
        cursor = connection.cursor()
        cursor.execute("SELECT 1")
        cursor.fetchall()
        cursor.close()

        connection.close()

        display_func.success(f"✅ Connection successful! Time: {end_time - start_time:.2f}s")
        return True
    except pyodbc.Error as e:
        display_func.error(f"❌ Connection failed: {e}")

        # Provide specific guidance based on error codes
        error_str = str(e)
        if "08001" in error_str:
            display_func.warning("Cannot reach the server. Check server name and firewall rules.")
        elif "28000" in error_str or "18456" in error_str:
            display_func.warning("Authentication failed. Check username and password.")
        elif "42000" in error_str:
            display_func.warning("Database access error. Check database name and permissions.")
        elif "01000" in error_str and ("TLS" in error_str or "SSL" in error_str):
            display_func.warning("SSL/TLS error. Try using TrustServerCertificate=yes.")
        elif "IM002" in error_str:
            display_func.warning("Driver not found. Check ODBC driver installation.")

        return False
    except Exception as e:
        display_func.error(f"❌ Unexpected error: {e}")
        return False


# Fix for debugging database connection issues
def debug_db_connection(server, database, username, password, use_entra=False, entra_domain=None):
    """
    Provides a comprehensive diagnostic of database connection issues by testing multiple
    connection string variations and driver options.

    Returns a dict with test results and recommendations.
    """
    results = {
        "success": False,
        "successful_conn_str": None,
        "tested_variations": [],
        "recommendation": ""
    }

    # Test different ODBC driver versions
    drivers_to_try = [
        "ODBC Driver 18 for SQL Server",
        "ODBC Driver 17 for SQL Server",
        "SQL Server Native Client 11.0",
        "SQL Server"  # Basic fallback
    ]

    # Test with and without port specification
    server_variations = [
        (f"{server},{DB_PORT}", "with port"),
        (server, "without port")
    ]

    # Test with and without TrustServerCertificate
    trust_cert_variations = [
        ("no", "with certificate validation"),
        ("yes", "without certificate validation")
    ]

    for driver in drivers_to_try:
        for server_var, server_desc in server_variations:
            for trust_cert, trust_desc in trust_cert_variations:
                if use_entra:
                    if not entra_domain:
                        continue
                    conn_str = (
                        f"DRIVER={{{driver}}};"
                        f"SERVER={server_var};"
                        f"DATABASE={database};"
                        f"Authentication=ActiveDirectoryPassword;"
                        f"UID={username}@{entra_domain};"
                        f"PWD={password};"
                        f"Encrypt=yes;TrustServerCertificate={trust_cert};Connection Timeout=30;"
                    )
                    test_desc = f"Entra Auth with {driver}, {server_desc}, {trust_desc}"
                else:
                    conn_str = (
                        f"DRIVER={{{driver}}};"
                        f"SERVER={server_var};"
                        f"DATABASE={database};"
                        f"UID={username};"
                        f"PWD={password};"
                        f"Encrypt=yes;TrustServerCertificate={trust_cert};Connection Timeout=30;"
                    )
                    test_desc = f"SQL Auth with {driver}, {server_desc}, {trust_desc}"

                # Test this variation
                try:
                    st.sidebar.text(f"Testing: {test_desc}")
                    start_time = time.time()
                    connection = pyodbc.connect(conn_str, timeout=15)  # Short timeout for testing
                    end_time = time.time()

                    # Try a simple query
                    cursor = connection.cursor()
                    cursor.execute("SELECT 1")
                    cursor.fetchall()
                    cursor.close()

                    connection.close()

                    results["tested_variations"].append({
                        "description": test_desc,
                        "success": True,
                        "time": f"{end_time - start_time:.2f}s"
                    })

                    # If this is our first success, save it
                    if not results["success"]:
                        results["success"] = True
                        results["successful_conn_str"] = conn_str
                        results["recommendation"] = f"Use {test_desc}"

                except Exception as e:
                    results["tested_variations"].append({
                        "description": test_desc,
                        "success": False,
                        "error": str(e)
                    })

    # If all tests failed, provide a comprehensive error analysis
    if not results["success"]:
        error_counts = {}
        for test in results["tested_variations"]:
            error_msg = test["error"]
            if error_msg in error_counts:
                error_counts[error_msg] += 1
            else:
                error_counts[error_msg] = 1

        most_common_error = max(error_counts.items(), key=lambda x: x[1])
        results["recommendation"] = analyze_db_error(most_common_error[0])

    return results


def analyze_db_error(error_msg):
    """Analyzes a database error message and returns recommendations."""
    if "08001" in error_msg:
        return ("Cannot reach the server. Check:\n"
                "1. Server name is correct\n"
                "2. Azure firewall allows your IP\n"
                "3. Network connectivity\n"
                "4. VPN/proxy settings if applicable")
    elif "28000" in error_msg or "18456" in error_msg:
        return ("Authentication failed. Check:\n"
                "1. Username and password are correct\n"
                "2. User exists in the database\n"
                "3. User has permission to access this database\n"
                "4. For Entra auth, verify domain and permissions")
    elif "42000" in error_msg:
        return ("Database access error. Check:\n"
                "1. Database name is correct\n"
                "2. User has permission to access this database\n"
                "3. Database exists on the server")
    elif "01000" in error_msg and ("TLS" in error_msg or "SSL" in error_msg):
        return ("SSL/TLS error. Try:\n"
                "1. Setting TrustServerCertificate=yes\n"
                "2. Updating your ODBC driver\n"
                "3. Installing required certificates")
    elif "IM002" in error_msg:
        return ("ODBC driver not found. Check:\n"
                "1. Install Microsoft ODBC Driver for SQL Server\n"
                "2. Try different driver versions (17, 18)\n"
                "3. Use SQL Server Native Client if available")
    elif "HYT00" in error_msg:
        return ("Connection timeout. Check:\n"
                "1. Server is reachable\n"
                "2. Increase connection timeout value\n"
                "3. Network latency issues")
    else:
        return (f"Unrecognized error: {error_msg}\n"
                "General recommendations:\n"
                "1. Verify server, database, and credential information\n"
                "2. Check firewall rules in Azure\n"
                "3. Test connection from another tool (e.g., SSMS)\n"
                "4. Check logs in Azure Portal")

if __name__ == "__main__":
    main()
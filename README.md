# DB Wizard Utils

Python scripts for database analysis, documentation, foreign key correction, and test data generation. Most tools leverage OpenAI's generative AI to provide intelligent analysis and suggestions.

## Available Scripts

- [Table Describer](#table-describer) - Generates detailed Markdown documentation for database tables using AI
- [Database Analyzer](#database-analyzer) - Creates visual dependency graphs of database relationships
- [Foreign Key Analyzer](#foreign-key-analyzer) - Identifies and suggests missing foreign key relationships
- [Generate Test Data](#generate-test-data) - Creates realistic test data while respecting database constraints
- [CSV to Database Import](#csv-to-database-import) - Imports CSV data into database tables while respecting foreign key constraints

## Prerequisites

Before using these scripts, ensure you have:

- Python 3.8 or higher installed
- Git (for cloning the repository)
- Access to a supported database (see Supported Databases section below)
- OpenAI API key (for AI-powered features)

## Supported Databases

The scripts should work with any database that SQLAlchemy supports, including:

- MySQL
- PostgreSQL
- SQLite
- Oracle
- Microsoft SQL Server

These scripts have been tested with MySQL, and the MySQL Python library is installed by default. 

To use a different database:

  1. Install the appropriate Python library (e.g. `psycopg2-binary` for PostgreSQL)
  2. Provide the appropriate SQLAlchemy connection string in the `DB_CONNECTION_STRING` environment variable. For example:

- MySQL: `mysql+pymysql://user:pass@host:port/db`
- PostgreSQL: `postgresql://user:pass@host:port/db`
- SQLite: `sqlite:///path/to/database.db`


## Development Environment Setup

### Environment variables

Required environment variables (in `.env`):

| Variable                  | Description                                                                                                            | Required | Default       | Used in Scripts |
| ------------------------- | ---------------------------------------------------------------------------------------------------------------------- | -------- | ------------- | --------------- |
| DB_CONNECTION_STRING      | SQLAlchemy connection string (e.g., "mysql+pymysql://user:pass@host:port/db" or "postgresql://user:pass@host:port/db") | Yes      | -             | All scripts     |
| OPENAI_API_KEY            | Your OpenAI API key                                                                                                    | Yes      | -             | FK Analyzer, Test Data Generator, Table Describer |
| AI_MODEL                  | OpenAI model to use                                                                                                    | No       | gpt-3.5-turbo | FK Analyzer, Test Data Generator, Table Describer |
| TABLES                    | Comma-separated list of tables to generate data for; can also be specified as a command line parameter                 | No       | All tables    | Test Data Generator |
| LOG_LEVEL                 | Logging verbosity level                                                                                                | No       | DEBUG         | All scripts except Database Analyzer |
| DB_DESCRIPTION            | Description of the database to enhance AI data generation                                                              | No       | -             | FK Analyzer, Test Data Generator, Table Describer |
| SAVE_SQL                  | Whether to save the generated SQL                                                                                      | No       | true          | Test Data Generator |
| DEFAULT_ROWS              | Number of rows to generate per table                                                                                   | No       | 10            | Test Data Generator |
| MAX_RETRY_ATTEMPTS        | Number of retry attempts if row generation fails                                                                       | No       | 3             | Test Data Generator |
| ROLLBACK_ON_TABLE_FAILURE | Whether to rollback all changes if any table fails                                                                     | No       | true          | Test Data Generator |
| CONFIDENCE_THRESHOLD      | Threshold for foreign key relationship confidence (0.0 to 1.0)                                                        | No       | 0.8           | FK Analyzer |
| IGNORED_COLUMNS          | Comma-separated list of columns to ignore in foreign key analysis                                                     | No       | -             | FK Analyzer |
| DB_INSTRUCTIONS          | Additional instructions for database analysis (e.g., naming conventions, framework details)                            | No       | -             | FK Analyzer, Table Describer |

### Set up Virtual Environment

Set up a virtual environment before running the scripts to avoid dependency version conflicts between projects:

```bash
# Create a new virtual environment
python -m venv .venv

# Activate the virtual environment
source .venv/bin/activate
```

**Note**:
You will need to active the virtual environment whenever you open a new Terminal window.

## Table Describer

A tool that automatically generates detailed Markdown documentation for database tables using AI. It analyzes table structures, relationships, and generates human-readable descriptions of each table's purpose and relationships.

### Run

The first time you run the script, please install the dependencies:

```bash
pip install -r requirements/table_describer.txt
```

Then run the script:

```bash
python table_describer.py
```

**Note**:
- Find the logs in `logs/table_descriptions_YYYYMMDD.log`
- The generated documentation will be saved in `output/table_descriptions_YYYYMMDD_HHMMSS.md`

## Database Analyzer

A tool that analyses database dependencies and generates visual dependency graphs. It helps understand table relationships and determines the correct order for data insertion. Works with any database supported by SQLAlchemy.

### Run

The first time you run the script, please install the dependencies:

```bash
pip install -r requirements/database_analyzer.txt
```

Then run the script:

```bash
python database_analyzer.py
```

Note: the dependency graph is saved as `output/database_dependencies.png`

## Foreign Key Analyzer

A tool that analyzes your database to identify potential foreign key relationships that haven't been explicitly defined. It uses AI to suggest appropriate foreign key constraints based on column naming patterns and existing relationships. 

The script has an interactive prompt to allow users to accept or reject suggestions. SQL statements for adding the accepted foreign key constraints are saved to a file so they can be manually applied to the database.

### Run

The first time you run the script, please install the dependencies:

```bash
pip install -r requirements/foreign_key_analyzer.txt
```

Then run the script:

```bash
python foreign_key_analyzer.py
```

**Note**:
- Find the logs in `logs/foreign_key_analysis_YYYYMMDD.log`
- The generated SQL statements will be saved in `output/foreign_key_definitions_YYYYMMDD_HHMMSS.sql`


## Generate Test Data

A tool that generates and inserts realistic test data for your database tables in the correct dependency order. It uses OpenAI's API to create contextually appropriate data while respecting foreign key constraints. Works with any database supported by SQLAlchemy.

### Run

The first time you run the script, please install the dependencies:

```bash
pip install -r requirements/generate_test_data.txt
```

Then run the script:

```bash
python generate_test_data.py
```

**Note**:

- Find the logs in `logs/test_data_generation_YYYYMMDD.log`
- If `SAVE_SQL` is set, the SQL inserts will be found in `output/test_data_inserts__YYYYMMDDHHSS.sql`

## CSV to Database Import

A tool that imports data from CSV files into a database. It respects foreign key and automatically determines the correct order for data insertion based on table dependencies. It generates SQL files for the inserts so that the test data can be replayed on another database, and supports multiple CSV files per table.

### Run

The first time you run the script, please install the dependencies:

```bash
pip install -r requirements/csv_to_db.txt
```

Then run the script with a folder containing your CSV files:

```bash
python csv_to_db.py /path/to/csv/folder
```

**Note**:
- CSV files should be named after their corresponding table names
- Multiple files for the same table can be numbered (e.g., booking1.csv, booking2.csv)
- The script will create an `output` directory containing SQL files for each table's inserts
- Make sure your `.env` file contains the `DB_CONNECTION_STRING` variable

## Running Tests

The project includes unit tests for the test data generation functionality. The tests cover:

- Database connection handling
- Configuration loading and validation
- Data generation with AI
- Error handling and retries
- Transaction management
- Logging setup

### Run Tests

First, make sure you have activated your virtual environment (see instructions above)

Then install the test dependencies:

```bash
pip install -r requirements/database_analyzer.txt
pip install -r requirements/test.txt
```

Run the tests using Python's unittest framework:

```bash
python -m unittest tests.py
```

The tests will run and display a summary of passed and failed tests. For more detailed output, you can use the `-v` flag:

```bash
python -m unittest -v tests.py
```

**Note**: 
- Some test failures are expected and are part of testing error handling scenarios. The final test summary should show "OK" if all tests passed as expected.
- The tests use mocking to avoid actual database connections or API calls, so they can be run without a live database or OpenAI API key.
- Make sure you're in your virtual environment (you should see `(.venv)` in your terminal prompt) before running the tests.